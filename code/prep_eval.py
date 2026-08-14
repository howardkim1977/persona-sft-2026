# -*- coding: utf-8 -*-
"""
prep_eval.py — 평가 세트 구축 (사전등록 v0.2 6절 준수)

H1: 중립 개방형 프롬프트 500 (과제 8종 균형, C0 생성에서 instruction만,
    학습 C0 instruction과 완전 중복 제거). eval_sets/h1_prompts.jsonl
H2: 평가 전용 페르소나(gen_main/eval_personas.jsonl, 학습과 uuid 분리)에서
    연령 축(60대+ vs 40세 미만) 각 100명, 지역 축(비수도권 vs 수도권) 각 100명
    층화 추출 → 페르소나 조건화 생성에서 instruction만 → 총 400문항.
    eval_sets/h2_items.jsonl
H3: KoBEST 5과제 전체 + KMMLU 카테고리 층화 2,000 (시드 42).
    eval_sets/h3_items.jsonl
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from smoke import load_key, call_model
from map_persona import map_persona
from tasks import TASK_TYPES, build_generation_prompt, parse_pair
from gen_main import hangul_ratio, MODEL_ID

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "eval_sets")
TASK_LIST = list(TASK_TYPES.keys())
METRO = ("서울특별시", "경기도", "인천광역시")


def gen_instruction(api_key, cond, task, persona_block=None, retries=3):
    """생성에서 instruction만 취득. 실패/필터 미달 시 None."""
    prompt = build_generation_prompt(cond, task, persona_block)
    for _ in range(retries):
        out = call_model(api_key, MODEL_ID, prompt)
        if "error" in out:
            continue
        pair = parse_pair(out.get("content", ""))
        if pair and len(pair["instruction"]) >= 20 \
                and hangul_ratio(pair["instruction"]) >= 0.5:
            return pair["instruction"]
    return None


def build_h2(api_key):
    ev = [json.loads(l) for l in open(os.path.join(BASE, "gen_main/eval_personas.jsonl"))]
    mapped = [map_persona(p) for p in ev]
    old = [m for m in mapped if (m["segments"]["연령대"] or 0) >= 7]
    young = [m for m in mapped if int(m["raw"]["age"]) < 40]
    nonmetro = [m for m in mapped if m["segments"]["시도"] not in METRO]
    metro = [m for m in mapped if m["segments"]["시도"] in METRO]
    print(f"평가 페르소나 풀: 60대+ {len(old)} / 40미만 {len(young)} / "
          f"비수도권 {len(nonmetro)} / 수도권 {len(metro)}")

    groups = [("age", "60plus", old, 43), ("age", "under40", young, 43),
              ("region", "nonmetro", nonmetro, 44), ("region", "metro", metro, 44)]
    items = []
    jobs = []
    for axis, gname, pool, seed in groups:
        rng = random.Random(seed)
        pool = pool[:]
        rng.shuffle(pool)
        sel, backup = pool[:100], pool[100:200]
        for i, m in enumerate(sel):
            jobs.append({"axis": axis, "group": gname, "rank": i,
                         "m": m, "backup": backup, "task": TASK_LIST[i % 8]})

    def work(job):
        ins = gen_instruction(api_key, "C2", job["task"],
                              job["m"]["persona_prompt"])
        m = job["m"]
        b = 0
        while ins is None and b < len(job["backup"]):
            m = job["backup"][b]
            ins = gen_instruction(api_key, "C2", job["task"], m["persona_prompt"])
            b += 1
        return {"axis": job["axis"], "group": job["group"],
                "task": job["task"], "uuid": m["uuid"],
                "segments": m["segments"], "instruction": ins,
                "replaced": b}

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            items.append(f.result())
            if n % 50 == 0:
                print(f"  H2 {n}/{len(jobs)}")
    ok = [i for i in items if i["instruction"]]
    with open(os.path.join(OUT, "h2_items.jsonl"), "w") as f:
        for i in ok:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"H2 완료: {len(ok)}/400 (대체 사용 {sum(1 for i in ok if i['replaced'])})")


def build_h1(api_key):
    train_c0 = set()
    for l in open(os.path.join(BASE, "gen_main/results.jsonl")):
        r = json.loads(l)
        if r["condition"] == "C0" and r["status"] == "valid":
            train_c0.add(r["pair"]["instruction"].strip())

    per_task = 66  # 8 x 66 = 528 생성 → 중복 제거 후 500 절단
    jobs = [(t, i) for t in TASK_LIST for i in range(per_task)]

    def work(job):
        t, i = job
        return t, gen_instruction(api_key, "C0", t)

    got = {t: [] for t in TASK_LIST}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            t, ins = f.result()
            if ins:
                got[t].append(ins)
            if n % 100 == 0:
                print(f"  H1 {n}/{len(jobs)}")

    seen = set(train_c0)
    final = []
    # 과제 균형: 앞 4과제 63, 뒤 4과제 62 = 500
    quota = {t: (63 if idx < 4 else 62) for idx, t in enumerate(TASK_LIST)}
    for t in TASK_LIST:
        uniq = []
        for ins in got[t]:
            k = ins.strip()
            if k not in seen:
                seen.add(k)
                uniq.append(ins)
        if len(uniq) < quota[t]:
            print(f"  경고: {t} 고유 프롬프트 {len(uniq)} < 할당 {quota[t]}")
        final.extend({"task": t, "prompt": p} for p in uniq[:quota[t]])
    with open(os.path.join(OUT, "h1_prompts.jsonl"), "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"H1 완료: {len(final)}/500")


def build_h3():
    rows = [json.loads(l) for l in
            open("/Users/howardkim/Research/ai-comparison/data/items_main.jsonl")]
    kobest = [r for r in rows if r["benchmark"] == "kobest"]
    kmmlu = [r for r in rows if r["benchmark"] == "kmmlu"]
    # KMMLU 카테고리 비례 층화 2,000 (시드 42)
    rng = random.Random(42)
    by_cat = {}
    for r in kmmlu:
        by_cat.setdefault(r["category"], []).append(r)
    total = len(kmmlu)
    picked = []
    leftovers = []
    for cat, items in sorted(by_cat.items()):
        k = int(len(items) / total * 2000)  # 내림 후 큰 잔여분부터 채움
        rng.shuffle(items)
        picked.extend(items[:k])
        frac = len(items) / total * 2000 - k
        leftovers.append((frac, cat, items[k:]))
    # 결정적 보정: 잔여 소수부가 큰 카테고리 순으로 1개씩 추가해 정확히 2,000
    leftovers.sort(key=lambda x: (-x[0], x[1]))
    li = 0
    while len(picked) < 2000 and li < len(leftovers):
        frac, cat, rest = leftovers[li]
        if rest:
            picked.append(rest[0])
        li += 1
    h3 = kobest + picked
    with open(os.path.join(OUT, "h3_items.jsonl"), "w") as f:
        for r in h3:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"H3 완료: kobest {len(kobest)} + kmmlu {len(picked)} = {len(h3)}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    key = load_key()
    build_h3()
    build_h2(key)
    build_h1(key)
    print("평가 세트 구축 완료")
