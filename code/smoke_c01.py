# -*- coding: utf-8 -*-
"""
smoke_c01.py — C0(무페르소나)/C1(임의 페르소나) 조건 스모크 (Phase 0)

C0: 반복 6회 x 과제 3종 x 모델 3종 = 54 (동일 프롬프트, temperature로 변동)
C1: 임의 페르소나 6명 x 과제 3종 x 모델 3종 = 54
결과: smoke/smoke_c01_results.jsonl (raw 전문 저장)
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from smoke import MODELS, TASKS, load_key, call_model
from tasks import build_generation_prompt, parse_pair, make_random_persona, \
    random_persona_block

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke")
SEED = 778
N_REP = 6


def main():
    key = load_key()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    c1_personas = [make_random_persona(rng) for _ in range(N_REP)]
    print("C1 임의 페르소나:")
    for p in c1_personas:
        print(f"  {p['sex']} {p['age']}세 {p['province']} / {p['occupation']} / {p['education_level']}")

    jobs = []
    for rep in range(N_REP):
        for task in TASKS:
            p0 = build_generation_prompt("C0", task)
            p1 = build_generation_prompt(
                "C1", task, random_persona_block(c1_personas[rep]))
            for mkey, mid in MODELS.items():
                jobs.append(("C0", rep, task, mkey, mid, p0, None))
                jobs.append(("C1", rep, task, mkey, mid, p1, c1_personas[rep]))

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(call_model, key, mid, prompt):
                (cond, rep, task, mkey, persona)
                for cond, rep, task, mkey, mid, prompt, persona in jobs}
        for fut in as_completed(futs):
            cond, rep, task, mkey, persona = futs[fut]
            out = fut.result()
            pair = parse_pair(out.get("content", "")) if "error" not in out else None
            results.append({
                "condition": cond, "rep": rep, "persona": persona,
                "task": task, "model": mkey,
                "ok": pair is not None, "pair": pair,
                "raw_content": out.get("content", ""),
                "finish_reason": out.get("finish_reason"),
                "usage": out.get("usage"), "error": out.get("error"),
            })
            print(f"  [{len(results)}/{len(jobs)}] {cond} {mkey} r{rep} {task} "
                  f"{'OK' if pair else 'FAIL'}")

    out_path = os.path.join(OUT_DIR, "smoke_c01_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== 요약 (조건 x 모델 형식 성공) ===")
    for cond in ("C0", "C1"):
        for mkey in MODELS:
            rs = [r for r in results if r["condition"] == cond and r["model"] == mkey]
            print(f"  {cond} {mkey}: {sum(r['ok'] for r in rs)}/{len(rs)}")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
