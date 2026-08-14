# -*- coding: utf-8 -*-
"""
smoke.py — 생성기 후보 한국어 품질 스모크 테스트 (Phase 0)

페르소나 6명(연령·지역 다양화) x 과제 3종 x 생성기 3종 = 54호출, 조건 C2.
결과는 smoke/smoke_results.jsonl 에 저장한다. 본 실험 아님(사전등록 전 파이프라인
검증 용도), 파라미터는 임시값.
"""

import csv
import json
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlreq

from map_persona import map_persona
from tasks import build_generation_prompt, parse_pair

NEMOTRON_CSV = os.path.expanduser(
    "~/Research/persona-validation/nemotron_personas_korea.csv")
ENV_PATH = os.path.expanduser("~/Research/ai-comparison/.env")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke")

MODELS = {
    "v4-flash": "deepseek/deepseek-v4-flash-0731",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "kimi-k3": "moonshotai/kimi-k3",
}
TASKS = ["정보질의", "조언요청", "글쓰기"]
SEED = 777
MAX_TOKENS = 2000
TEMPERATURE = 1.0


def load_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("OpenRouter 키를 찾을 수 없습니다")


def pick_personas(n_scan: int = 120000) -> list:
    """연령대·지역이 다양하도록 6명 선정(시드 고정).
    목표 셀: 70대이상x2(수도권/비수도권), 50-60대x2, 20-30대x2. 성별 균형."""
    rng = random.Random(SEED)
    buckets = {"old": [], "mid": [], "young": []}
    with open(NEMOTRON_CSV) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= n_scan:
                break
            age = int(row["age"])
            key = "old" if age >= 70 else ("mid" if age >= 50 else
                                           ("young" if age <= 39 else None))
            if key and len(buckets[key]) < 400:
                buckets[key].append(row)
    chosen = []
    # old: 비수도권 1 + 수도권 1
    non_capital = [p for p in buckets["old"] if p["province"] not in ("서울", "경기", "인천")]
    capital = [p for p in buckets["old"] if p["province"] in ("서울", "경기", "인천")]
    chosen.append(rng.choice(non_capital))
    chosen.append(rng.choice(capital))
    for key in ("mid", "young"):
        males = [p for p in buckets[key] if p["sex"] == "남자"]
        females = [p for p in buckets[key] if p["sex"] == "여자"]
        chosen.append(rng.choice(males))
        chosen.append(rng.choice(females))
    return chosen


def call_model(key: str, model_id: str, prompt: str, retries: int = 3) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if model_id.startswith(("moonshotai/", "deepseek/")):
        payload["reasoning"] = {"enabled": False}
    elif model_id.startswith("openai/gpt-oss"):
        payload["reasoning"] = {"effort": "low"}
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urlreq.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urlreq.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            choice = data["choices"][0]
            return {
                "content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage", {}),
            }
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt * 2)


def main():
    key = load_key()
    os.makedirs(OUT_DIR, exist_ok=True)
    personas = pick_personas()
    mapped = [map_persona(p) for p in personas]
    print("선정 페르소나:")
    for m in mapped:
        s = m["segments"]
        print(f"  {s['성별']} {m['raw']['age']}세 {s['시도']} / {s['직업']}")

    jobs = []
    for pi, m in enumerate(mapped):
        for task in TASKS:
            prompt = build_generation_prompt("C2", task, m["persona_prompt"])
            for mkey, mid in MODELS.items():
                jobs.append((pi, task, mkey, mid, prompt))

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(call_model, key, mid, prompt): (pi, task, mkey)
                for pi, task, mkey, mid, prompt in jobs}
        for fut in as_completed(futs):
            pi, task, mkey = futs[fut]
            out = fut.result()
            pair = parse_pair(out.get("content", "")) if "error" not in out else None
            results.append({
                "persona_idx": pi,
                "segments": mapped[pi]["segments"],
                "task": task, "model": mkey,
                "ok": pair is not None,
                "pair": pair,
                "raw_content": out.get("content", "")[:2000],
                "finish_reason": out.get("finish_reason"),
                "usage": out.get("usage"),
                "error": out.get("error"),
            })
            print(f"  [{len(results)}/{len(jobs)}] {mkey} p{pi} {task} "
                  f"{'OK' if pair else 'FAIL'}")

    out_path = os.path.join(OUT_DIR, "smoke_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 요약
    print("\n=== 요약 ===")
    for mkey in MODELS:
        rs = [r for r in results if r["model"] == mkey]
        ok = sum(r["ok"] for r in rs)
        toks = [r["usage"]["completion_tokens"] for r in rs
                if r.get("usage") and r["usage"].get("completion_tokens")]
        avg = sum(toks) / len(toks) if toks else 0
        errs = sum(1 for r in rs if r.get("error"))
        print(f"  {mkey}: 형식 성공 {ok}/{len(rs)}, 오류 {errs}, "
              f"평균 출력 {avg:.0f}tok")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
