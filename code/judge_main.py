# -*- coding: utf-8 -*-
"""
judge_main.py — H2 하위집단 응답 4심판 채점 (사전등록 v0.2 6.2절)

대상: eval_out/h2_{model}.jsonl (7모델 x 400문항 = 2,800응답)
심판: 주심판 gpt-5.6-sol, claude-opus-5 (확증) / 보조 gpt-5.2, HCX-007 (탐색적)
척도: content(0-3, 확증), register(0-3, 탐색적). temperature 0.
HCX-007은 CLOVA Studio OpenAI 호환 엔드포인트, 60QPM 스로틀, 추론 비활성.
증분 저장·재개: judge_out/scores.jsonl
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlreq

from smoke import load_key  # OpenRouter 키

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE, "eval_sets")
OUT_DIR = os.path.join(BASE, "judge_out")
RESP_DIR = os.path.join(BASE, "eval_out")
ENV_CLOVA = os.path.expanduser("~/Research/ai-comparison/.env")

MODELS = ["base", "C0_s42", "C0_s777", "C1_s42", "C1_s777", "C2_s42", "C2_s777"]
JUDGES = {
    "sol": {"provider": "openrouter", "id": "openai/gpt-5.6-sol", "role": "primary"},
    "opus5": {"provider": "openrouter", "id": "anthropic/claude-opus-5", "role": "primary"},
    "gpt52": {"provider": "openrouter", "id": "openai/gpt-5.2", "role": "aux"},
    "hcx": {"provider": "clova", "id": "HCX-007", "role": "aux"},
}
CLOVA_BASE = "https://clovastudio.stream.ntruss.com/v1/openai"
CLOVA_QPM = int(os.environ.get("CLOVA_QPM", "60"))
_clova_lock = threading.Lock()
_clova_last = [0.0]

JUDGE_TMPL = """당신은 한국어 AI 어시스턴트 응답의 품질 평가자입니다. 아래 [화자 정보]를 가진 사용자가 보낸 [질의]와, AI 어시스턴트의 [응답]을 평가하십시오.

[화자 정보]
{speaker}

[질의]
{instruction}

[응답]
{response}

평가 기준 (각 0~3점):
- content: 응답이 질의의 상황과 요구에 충실하고 유용한가 (완전히 충실하고 유용 3 / 대체로 충실하나 일부 미흡 2 / 부분적으로만 유용 1 / 불충실하거나 유해·무관 0)
- register: 응답의 어휘·존대·설명 방식이 화자의 연령과 배경에 적절한가 (매우 적절 3 / 대체로 적절 2 / 부분적으로 부적절 1 / 명백히 부적절 0)

반드시 아래 JSON만 출력하십시오.
{{"content": <0-3>, "register": <0-3>, "reason": "<한 문장>"}}"""


def load_clova_key():
    with open(ENV_CLOVA) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith("CLOVA_STUDIO_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("CLOVA_STUDIO_API_KEY 없음")


def clova_throttle():
    interval = 60.0 / CLOVA_QPM * 1.05
    with _clova_lock:
        wait = _clova_last[0] + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _clova_last[0] = time.time()


def call_judge(jkey, prompt, keys, retries=4):
    cfg = JUDGES[jkey]
    for attempt in range(retries):
        try:
            if cfg["provider"] == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                body = {"model": cfg["id"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "max_tokens": 2000}
                auth = keys["openrouter"]
            else:
                clova_throttle()
                url = f"{CLOVA_BASE}/chat/completions"
                body = {"model": cfg["id"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "max_completion_tokens": 700,
                        "reasoning_effort": "none"}
                auth = keys["clova"]
            req = urlreq.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {auth}",
                                          "Content-Type": "application/json"})
            with urlreq.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"].get("content") or ""
        except Exception as e:
            if attempt == retries - 1:
                return f"__ERROR__{e}"
            time.sleep(2 ** attempt * 2)


def parse_score(text):
    if not text or text.startswith("__ERROR__"):
        return None
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        obj = json.loads(text[s:e + 1], strict=False)
        c, r = int(obj["content"]), int(obj["register"])
        if 0 <= c <= 3 and 0 <= r <= 3:
            return {"content": c, "register": r,
                    "reason": str(obj.get("reason", ""))[:300]}
    except Exception:
        pass
    return None


def speaker_block(seg):
    return (f"성별: {seg['성별']} / 연령대: {seg['연령대_라벨']} / "
            f"거주지: {seg['시도']} / 직업: {seg['직업']} / 학력: {seg['교육수준']}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = {"openrouter": load_key(), "clova": load_clova_key()}
    items = [json.loads(l) for l in open(os.path.join(EVAL_DIR, "h2_items.jsonl"))]

    responses = {}
    for m in MODELS:
        p = os.path.join(RESP_DIR, f"h2_{m}.jsonl")
        if not os.path.exists(p):
            print(f"경고: {p} 없음 — eval_all.sh 완료 후 실행하십시오")
            continue
        for l in open(p):
            r = json.loads(l)
            responses[(m, r["key"])] = r["response"]

    out_path = os.path.join(OUT_DIR, "scores.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["model"], r["item"], r["judge"]))

    jobs = []
    for i, it in enumerate(items):
        ikey = f"h2:{i}"
        for m in MODELS:
            resp = responses.get((m, ikey))
            if resp is None:
                continue
            prompt = JUDGE_TMPL.format(speaker=speaker_block(it["segments"]),
                                       instruction=it["instruction"],
                                       response=resp)
            for jkey in JUDGES:
                if (m, ikey, jkey) not in done:
                    jobs.append((m, ikey, jkey, it["axis"], it["group"], prompt))
    print(f"채점 작업 {len(jobs)}건 (기존 {len(done)}건 재사용)")

    t0 = time.time()
    n = 0
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(call_judge, j[2], j[5], keys): j for j in jobs}
            for fut in as_completed(futs):
                m, ikey, jkey, axis, group, _ = futs[fut]
                raw = fut.result()
                score = parse_score(raw)
                fout.write(json.dumps({
                    "model": m, "item": ikey, "judge": jkey,
                    "axis": axis, "group": group,
                    "score": score,
                    "raw": None if score else (raw or "")[:300],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
                if n % 200 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{len(jobs)} ({el:.0f}s, {n/el*60:.0f}/min)",
                          flush=True)
    print(f"완료: {out_path}")


if __name__ == "__main__":
    main()
