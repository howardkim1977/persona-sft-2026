# -*- coding: utf-8 -*-
"""
judge_smoke.py — C2 스모크 산출물 심판 채점 (생성기 잠정 선정용)

54건 x 심판 2인(gpt-5.6-sol, claude-opus-5). 척도(각 0-2):
  fidelity: instruction이 인물의 인구통계·배경·말투에 자연스럽게 부합하는가
  quality : response가 요청에 충실하고 정확하며 한국어가 자연스러운가
결과: smoke/judge_results.jsonl + 모델별 집계 출력.
temperature 0. 본 실험 심판 프로토콜의 축소 예행이다.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from smoke import load_key, pick_personas, MODELS
from map_persona import map_persona
from tasks import parse_pair
import smoke as smoke_mod
from urllib import request as urlreq
import time

JUDGES = {
    "sol": "openai/gpt-5.6-sol",
    "opus": "anthropic/claude-opus-5",
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke")

JUDGE_TMPL = """당신은 한국어 학습 데이터 품질 평가자입니다. 아래 [인물 정보]를 가진 가상 인물이 AI 어시스턴트에게 했다고 상정된 [요청]과 그에 대한 [응답]을 평가하십시오.

[인물 정보]
{persona}

[과제 유형] {task}

[요청(instruction)]
{instruction}

[응답(response)]
{response}

평가 기준 (각 0, 1, 2점):
- fidelity: 요청이 이 인물에게서 자연스럽게 나올 법한가 (내용이 인물의 삶·배경과 맞고, 말투·어휘가 연령·배경에 부합하면 2점 / 부분적이면 1점 / 인물과 무관하거나 부자연스러우면 0점)
- quality: 응답이 요청에 충실하고 정보가 적절하며 한국어가 자연스러운가 (충실하고 자연스러우면 2점 / 부분적 결함 1점 / 불충실하거나 부자연 0점)

반드시 아래 JSON만 출력하십시오.
{{"fidelity": <0|1|2>, "quality": <0|1|2>, "reason": "<한 문장>"}}"""


def call_judge(key: str, model_id: str, prompt: str, retries: int = 3) -> dict:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000,
    }).encode()
    for attempt in range(retries):
        try:
            req = urlreq.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urlreq.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            return {"content": msg.get("content") or "", "usage": data.get("usage", {})}
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt * 2)


def parse_score(text: str):
    if not text:
        return None
    s = text.find("{"); e = text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        obj = json.loads(text[s:e + 1], strict=False)
        f, q = int(obj["fidelity"]), int(obj["quality"])
        if f in (0, 1, 2) and q in (0, 1, 2):
            return {"fidelity": f, "quality": q, "reason": obj.get("reason", "")}
    except Exception:
        pass
    return None


def main():
    key = load_key()
    rows = [json.loads(l) for l in
            open(os.path.join(OUT_DIR, "smoke_results.jsonl"))]
    personas = [map_persona(p) for p in pick_personas()]

    items = []
    for r in rows:
        pair = r["pair"] or parse_pair(r["raw_content"])
        if not pair:
            continue
        items.append({
            "persona_idx": r["persona_idx"], "task": r["task"],
            "model": r["model"], "pair": pair,
        })
    print(f"채점 대상 {len(items)}건 x 심판 {len(JUDGES)}인")

    jobs = []
    for idx, it in enumerate(items):
        prompt = JUDGE_TMPL.format(
            persona=personas[it["persona_idx"]]["persona_prompt"],
            task=it["task"],
            instruction=it["pair"]["instruction"],
            response=it["pair"]["response"])
        for jkey, jid in JUDGES.items():
            jobs.append((idx, jkey, jid, prompt))

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(call_judge, key, jid, prompt): (idx, jkey)
                for idx, jkey, jid, prompt in jobs}
        for fut in as_completed(futs):
            idx, jkey = futs[fut]
            out = fut.result()
            score = parse_score(out.get("content", "")) if "error" not in out else None
            results.append({
                "item_idx": idx, "judge": jkey,
                "model": items[idx]["model"], "task": items[idx]["task"],
                "persona_idx": items[idx]["persona_idx"],
                "score": score, "usage": out.get("usage"),
                "error": out.get("error"),
                "raw": (out.get("content") or "")[:500],
            })
            print(f"  [{len(results)}/{len(jobs)}] {jkey} item{idx} "
                  f"{'OK' if score else 'FAIL'}")

    out_path = os.path.join(OUT_DIR, "judge_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== 모델별 평균 (심판 2인 평균) ===")
    for mkey in MODELS:
        rs = [r for r in results if r["model"] == mkey and r["score"]]
        if not rs:
            continue
        fid = sum(r["score"]["fidelity"] for r in rs) / len(rs)
        qua = sum(r["score"]["quality"] for r in rs) / len(rs)
        print(f"  {mkey}: fidelity {fid:.2f} / quality {qua:.2f} (n={len(rs)})")
    print("\n=== 심판별 x 모델 ===")
    for jkey in JUDGES:
        for mkey in MODELS:
            rs = [r for r in results
                  if r["judge"] == jkey and r["model"] == mkey and r["score"]]
            if not rs:
                continue
            fid = sum(r["score"]["fidelity"] for r in rs) / len(rs)
            qua = sum(r["score"]["quality"] for r in rs) / len(rs)
            print(f"  {jkey} x {mkey}: fidelity {fid:.2f} / quality {qua:.2f}")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
