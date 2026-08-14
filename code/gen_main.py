# -*- coding: utf-8 -*-
"""
gen_main.py — 본 실험 생성 (사전등록 v0.2 준수, OSF osf.io/fc8mn)

조건당 10,000쌍 x 3조건. 생성기 deepseek/deepseek-v4-flash-0731,
temperature 1.0, max_tokens 2000, 추론 비활성, 재생성 최대 3회.
품질 필터(조건 공통): instruction >= 20자, response >= 50자, 한글 비율 >= 50%.
필터 탈락·3회 실패는 결측 기록 후 대체 표본으로 보충(발생률 보고).

C2 표본 설계 (시드 42, 단일 패스 저수지 13,000):
  셔플 후 [0:10000] 학습, [10000:11000] 학습 보충 예비,
  [11000:13000] 평가 전용(uuid 분리) → gen_main/eval_personas.jsonl 저장.

재개 지원: gen_main/results.jsonl 에 증분 기록, 완료 키는 건너뜀.
실행: python3 gen_main.py [--dry N]
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from smoke import NEMOTRON_CSV, load_key, call_model
from map_persona import map_persona
from tasks import TASK_TYPES, build_generation_prompt, parse_pair, \
    make_random_persona, random_persona_block

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_main")
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
N_PER_COND = 10000
RESERVOIR_K = 13000
SEED = 42
WORKERS = 32
MAX_ATTEMPTS = 3
MAX_BACKFILL_ROUNDS = 3

TASK_LIST = list(TASK_TYPES.keys())  # 8종, 순서 고정


def hangul_ratio(s: str) -> float:
    if not s:
        return 0.0
    core = [c for c in s if not c.isspace()]
    if not core:
        return 0.0
    h = sum(1 for c in core if '가' <= c <= '힣')
    return h / len(core)


def passes_filter(pair: dict) -> bool:
    ins, res = pair["instruction"], pair["response"]
    if len(ins) < 20 or len(res) < 50:
        return False
    if hangul_ratio(ins) < 0.5 or hangul_ratio(res) < 0.5:
        return False
    return True


def task_for(i: int) -> str:
    return TASK_LIST[i % len(TASK_LIST)]


def build_manifest():
    """결정적 작업 목록 + C2 표본 분할을 생성(또는 기존 것 로드)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    man_path = os.path.join(OUT_DIR, "manifest.json")
    c2_train_path = os.path.join(OUT_DIR, "c2_train_personas.jsonl")
    c2_reserve_path = os.path.join(OUT_DIR, "c2_reserve_personas.jsonl")
    eval_path = os.path.join(OUT_DIR, "eval_personas.jsonl")
    c1_path = os.path.join(OUT_DIR, "c1_personas.jsonl")

    if os.path.exists(man_path):
        print("기존 manifest 재사용")
        return

    print(f"C2 저수지 표본 추출 (k={RESERVOIR_K}, seed={SEED}) — 100만 행 단일 패스...")
    rng = random.Random(SEED)
    sample = []
    with open(NEMOTRON_CSV) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if len(sample) < RESERVOIR_K:
                sample.append(row)
            else:
                j = rng.randint(0, i)
                if j < RESERVOIR_K:
                    sample[j] = row
    rng.shuffle(sample)
    train, reserve, ev = (sample[:N_PER_COND],
                          sample[N_PER_COND:N_PER_COND + 1000],
                          sample[N_PER_COND + 1000:])
    for path, part in ((c2_train_path, train), (c2_reserve_path, reserve),
                       (eval_path, ev)):
        with open(path, "w") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rng1 = random.Random(SEED + 1)
    c1 = [make_random_persona(rng1) for _ in range(N_PER_COND + 1000)]
    with open(c1_path, "w") as f:
        for p in c1:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    json.dump({
        "model": MODEL_ID, "n_per_cond": N_PER_COND, "seed": SEED,
        "reservoir_k": RESERVOIR_K, "tasks": TASK_LIST,
        "params": {"temperature": 1.0, "max_tokens": 2000,
                   "reasoning": "disabled"},
        "created": "2026-08-04",
        "prereg": "OSF project fc8mn, Open-Ended Registration v0.2",
    }, open(man_path, "w"), ensure_ascii=False, indent=2)
    print(f"manifest 생성: train {len(train)} / reserve {len(reserve)} / "
          f"eval {len(ev)} / c1 {len(c1)}")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def make_job(cond: str, idx: int, replace_no: int,
             c2_train, c2_reserve, c1_personas):
    """작업 1건 구성. replace_no>0 이면 보충 표본 사용."""
    task = task_for(idx)
    key = f"{cond}:{idx}:{replace_no}"
    if cond == "C0":
        prompt = build_generation_prompt("C0", task)
        meta = {}
    elif cond == "C1":
        p = (c1_personas[idx] if replace_no == 0
             else c1_personas[N_PER_COND + (idx + replace_no * 37) % 1000])
        prompt = build_generation_prompt("C1", task, random_persona_block(p))
        meta = {"persona": p}
    else:
        row = (c2_train[idx] if replace_no == 0
               else c2_reserve[(idx + replace_no * 37) % 1000])
        m = map_persona(row)
        prompt = build_generation_prompt("C2", task, m["persona_prompt"])
        meta = {"uuid": m["uuid"], "segments": m["segments"]}
    return {"key": key, "condition": cond, "idx": idx,
            "replace_no": replace_no, "task": task,
            "prompt": prompt, "meta": meta}


def gen_one(api_key: str, job: dict) -> dict:
    attempts = 0
    last = {}
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        out = call_model(api_key, MODEL_ID, job["prompt"])
        last = out
        pair = parse_pair(out.get("content", "")) if "error" not in out else None
        if pair:
            ok_filter = passes_filter(pair)
            return {"key": job["key"], "condition": job["condition"],
                    "idx": job["idx"], "replace_no": job["replace_no"],
                    "task": job["task"], **job["meta"],
                    "status": "valid" if ok_filter else "filtered",
                    "pair": pair, "attempts": attempts,
                    "usage": out.get("usage")}
    return {"key": job["key"], "condition": job["condition"],
            "idx": job["idx"], "replace_no": job["replace_no"],
            "task": job["task"], **job["meta"],
            "status": "failed", "pair": None, "attempts": attempts,
            "error": last.get("error"),
            "finish_reason": last.get("finish_reason")}


def run_jobs(api_key, jobs, results_path, t0):
    done = 0
    with open(results_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(gen_one, api_key, j) for j in jobs]
            for fut in as_completed(futs):
                r = fut.result()
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % 200 == 0:
                    el = time.time() - t0
                    rate = done / el * 60
                    print(f"  {done}/{len(jobs)} ({el:.0f}s, {rate:.0f}/min)",
                          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    api_key = load_key()
    build_manifest()
    c2_train = load_jsonl(os.path.join(OUT_DIR, "c2_train_personas.jsonl"))
    c2_reserve = load_jsonl(os.path.join(OUT_DIR, "c2_reserve_personas.jsonl"))
    c1_personas = load_jsonl(os.path.join(OUT_DIR, "c1_personas.jsonl"))
    results_path = os.path.join(OUT_DIR, "results.jsonl")

    n = args.dry if args.dry else N_PER_COND

    # 1) 기본 라운드: 미완료 키만
    existing = {}
    if os.path.exists(results_path):
        for r in load_jsonl(results_path):
            existing[r["key"]] = r
    jobs = []
    for cond in ("C0", "C1", "C2"):
        for i in range(n):
            key = f"{cond}:{i}:0"
            if key not in existing:
                jobs.append(make_job(cond, i, 0, c2_train, c2_reserve,
                                     c1_personas))
    print(f"기본 라운드: {len(jobs)}건 (기존 {len(existing)}건 재사용)", flush=True)
    if jobs:
        run_jobs(api_key, jobs, results_path, t0)

    # 2) 보충 라운드: 조건별 유효 수가 목표에 미달하면 대체 표본 생성
    for round_no in range(1, MAX_BACKFILL_ROUNDS + 1):
        rows = load_jsonl(results_path)
        need = {}
        for cond in ("C0", "C1", "C2"):
            valid = sum(1 for r in rows
                        if r["condition"] == cond and r["status"] == "valid")
            if valid < n:
                need[cond] = n - valid
        if not need:
            break
        print(f"보충 라운드 {round_no}: {need}", flush=True)
        done_keys = {r["key"] for r in rows}
        jobs = []
        for cond, cnt in need.items():
            bad_idx = [r["idx"] for r in rows
                       if r["condition"] == cond and r["replace_no"] == round_no - 1
                       and r["status"] != "valid"][:cnt] if round_no > 1 else \
                      [r["idx"] for r in rows
                       if r["condition"] == cond and r["replace_no"] == 0
                       and r["status"] != "valid"][:cnt]
            for i in bad_idx:
                key = f"{cond}:{i}:{round_no}"
                if key not in done_keys:
                    jobs.append(make_job(cond, i, round_no, c2_train,
                                         c2_reserve, c1_personas))
        if not jobs:
            break
        run_jobs(api_key, jobs, results_path, t0)

    # 3) 최종 요약
    rows = load_jsonl(results_path)
    pin, pout = 0.09 / 1e6, 0.18 / 1e6
    total_cost = 0.0
    print("\n=== 본 생성 요약 ===")
    for cond in ("C0", "C1", "C2"):
        rs = [r for r in rows if r["condition"] == cond]
        valid = sum(1 for r in rs if r["status"] == "valid")
        filt = sum(1 for r in rs if r["status"] == "filtered")
        fail = sum(1 for r in rs if r["status"] == "failed")
        i_tok = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rs)
        o_tok = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rs)
        cost = i_tok * pin + o_tok * pout
        total_cost += cost
        print(f"  {cond}: 유효 {valid} / 필터 탈락 {filt} / 실패 {fail} "
              f"(호출 {len(rs)}), 비용 ${cost:.2f}")
    print(f"  총비용 ${total_cost:.2f}, 총 소요 {(time.time()-t0)/60:.0f}분")
    print(f"저장: {results_path}")


if __name__ == "__main__":
    main()
