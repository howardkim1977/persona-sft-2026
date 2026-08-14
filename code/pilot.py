# -*- coding: utf-8 -*-
"""
pilot.py — 파일럿 생성: 조건당 200쌍 x 3조건 (Phase 0)

생성기: DeepSeek-V4-Flash-0731 (심판 채점 잠정 선정, 2026-08-03).
과제 8종을 조건당 25쌍씩 회전 배정. 형식 실패 시 최대 3회 재생성.
C2 표본: 전체 CSV 단일 패스 균등 저수지 표본 200 (원 데이터가 실분포이므로
균등 추출 = 실분포 비례). C1: 균등 무작위 조합 200. C0: 반복 200.
결과: pilot/pilot_pairs.jsonl (전문 저장) + 요약 통계.
"""

import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from smoke import NEMOTRON_CSV, load_key, call_model
from map_persona import map_persona
from tasks import TASK_TYPES, build_generation_prompt, parse_pair, \
    make_random_persona, random_persona_block

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot")
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
N_PER_COND = 200
SEED = 779
MAX_ATTEMPTS = 3
WORKERS = 16


def reservoir_sample(k: int, seed: int) -> list:
    rng = random.Random(seed)
    sample = []
    with open(NEMOTRON_CSV) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if len(sample) < k:
                sample.append(row)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = row
    return sample


def gen_one(key: str, job: dict) -> dict:
    attempts = 0
    last = {}
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        out = call_model(key, MODEL_ID, job["prompt"])
        last = out
        pair = parse_pair(out.get("content", "")) if "error" not in out else None
        if pair:
            return {**job["meta"], "ok": True, "pair": pair,
                    "attempts": attempts, "usage": out.get("usage"),
                    "finish_reason": out.get("finish_reason")}
    return {**job["meta"], "ok": False, "pair": None, "attempts": attempts,
            "usage": last.get("usage"), "finish_reason": last.get("finish_reason"),
            "error": last.get("error"),
            "raw_content": (last.get("content") or "")[:3000]}


def main():
    t0 = time.time()
    key = load_key()
    os.makedirs(OUT_DIR, exist_ok=True)
    task_list = list(TASK_TYPES.keys())  # 8종
    rng = random.Random(SEED)

    print("C2 저수지 표본 추출 중 (100만 행 단일 패스)...")
    c2_rows = reservoir_sample(N_PER_COND, SEED)
    c2_mapped = [map_persona(p) for p in c2_rows]
    c1_personas = [make_random_persona(rng) for _ in range(N_PER_COND)]

    jobs = []
    for i in range(N_PER_COND):
        task = task_list[i % len(task_list)]
        jobs.append({"prompt": build_generation_prompt("C0", task),
                     "meta": {"condition": "C0", "idx": i, "task": task}})
        jobs.append({"prompt": build_generation_prompt(
                        "C1", task, random_persona_block(c1_personas[i])),
                     "meta": {"condition": "C1", "idx": i, "task": task,
                              "persona": c1_personas[i]}})
        jobs.append({"prompt": build_generation_prompt(
                        "C2", task, c2_mapped[i]["persona_prompt"]),
                     "meta": {"condition": "C2", "idx": i, "task": task,
                              "uuid": c2_mapped[i]["uuid"],
                              "segments": c2_mapped[i]["segments"]}})

    print(f"총 {len(jobs)}건 생성 시작 (동시성 {WORKERS})")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(gen_one, key, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {n}/{len(jobs)} ({elapsed:.0f}s)")

    out_path = os.path.join(OUT_DIR, "pilot_pairs.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 요약 ---
    print(f"\n총 소요 {time.time()-t0:.0f}s")
    print("=== 조건별 요약 ===")
    pin = 0.09 / 1e6
    pout = 0.18 / 1e6
    total_cost = 0.0
    for cond in ("C0", "C1", "C2"):
        rs = [r for r in results if r["condition"] == cond]
        ok = [r for r in rs if r["ok"]]
        retried = sum(1 for r in rs if r["attempts"] > 1)
        i_tok = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rs)
        o_tok = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rs)
        cost = i_tok * pin + o_tok * pout
        total_cost += cost
        ins_len = [len(r["pair"]["instruction"]) for r in ok]
        res_len = [len(r["pair"]["response"]) for r in ok]
        # distinct-2 (instruction, 공백 토큰 기준)
        bigrams = set(); n_bi = 0
        for r in ok:
            toks = r["pair"]["instruction"].split()
            for a, b in zip(toks, toks[1:]):
                bigrams.add((a, b)); n_bi += 1
        d2 = len(bigrams) / n_bi if n_bi else 0
        print(f"  {cond}: 성공 {len(ok)}/{len(rs)} (재시도 {retried}), "
              f"instr 평균 {sum(ins_len)/max(len(ins_len),1):.0f}자, "
              f"resp 평균 {sum(res_len)/max(len(res_len),1):.0f}자, "
              f"distinct-2 {d2:.3f}, 비용 ${cost:.3f}")
    print(f"  파일럿 생성 총비용: ${total_cost:.3f}")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
