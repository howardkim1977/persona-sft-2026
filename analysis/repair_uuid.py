# -*- coding: utf-8 -*-
"""
repair_uuid.py — 산출물의 결측 uuid 메타데이터 사후 복구 (2026-08-06)

원인: Nemotron CSV의 UTF-8 BOM 때문에 첫 컬럼 키가 '﻿uuid'로 읽혀
map_persona가 uuid를 None으로 반환했다. 실험 통제(학습/평가 페르소나 분리)는
리스트 분할로 이루어졌으므로 영향이 없으며(분리 검증 완료: 교집합 0),
결측은 추적성 메타데이터에 한정된다.

복구 방식: 생성 시 사용한 결정적 선택 규칙을 그대로 재현한다.
  C2 생성: replace_no=0 → c2_train[idx], r>0 → c2_reserve[(idx+r*37)%1000]
  H2 문항: 축·집단별 시드 고정 셔플 상위 100명에서 segments 완전 일치로 대응
출력: *_uuid.jsonl (원본은 보존)
"""

import json
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from map_persona import map_persona  # noqa: E402

BOM_KEY = "﻿uuid"


def uid_of(row):
    return row.get("uuid") or row.get(BOM_KEY)


def load(p):
    return [json.loads(l) for l in open(os.path.join(BASE, p))]


def repair_generation():
    train = load("gen_main/c2_train_personas.jsonl")
    reserve = load("gen_main/c2_reserve_personas.jsonl")
    rows = load("gen_main/results.jsonl")
    fixed = 0
    for r in rows:
        if r["condition"] != "C2":
            continue
        i, rn = r["idx"], r["replace_no"]
        src = train[i] if rn == 0 else reserve[(i + rn * 37) % 1000]
        # 무결성 확인: 저장된 segments와 원본 매핑이 일치해야 한다
        seg = map_persona(src)["segments"]
        if r.get("segments") and r["segments"] != seg:
            raise RuntimeError(f"segments 불일치: {r['key']}")
        r["uuid"] = uid_of(src)
        fixed += 1
    out = os.path.join(BASE, "gen_main/results_uuid.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"생성 결과: C2 {fixed}건 uuid 복구 → {out}")


def repair_h2():
    ev = [map_persona(r) for r in load("gen_main/eval_personas.jsonl")]
    raw = load("gen_main/eval_personas.jsonl")
    METRO = ("서울특별시", "경기도", "인천광역시")
    old = [i for i, m in enumerate(ev) if (m["segments"]["연령대"] or 0) >= 7]
    young = [i for i, m in enumerate(ev) if int(m["raw"]["age"]) < 40]
    nonmetro = [i for i, m in enumerate(ev) if m["segments"]["시도"] not in METRO]
    metro = [i for i, m in enumerate(ev) if m["segments"]["시도"] in METRO]
    groups = {("age", "60plus"): (old, 43), ("age", "under40"): (young, 43),
              ("region", "nonmetro"): (nonmetro, 44),
              ("region", "metro"): (metro, 44)}
    pools = {}
    for key, (pool, seed) in groups.items():
        p = pool[:]
        random.Random(seed).shuffle(p)
        pools[key] = p[:200]          # 선정 100 + 예비 100

    items = load("eval_sets/h2_items.jsonl")
    fixed = amb = 0
    for it in items:
        cands = [i for i in pools[(it["axis"], it["group"])]
                 if ev[i]["segments"] == it["segments"]]
        if len(cands) == 1:
            it["uuid"] = uid_of(raw[cands[0]])
            fixed += 1
        else:
            amb += 1
    out = os.path.join(BASE, "eval_sets/h2_items_uuid.jsonl")
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"H2 문항: {fixed}/{len(items)} 복구, 모호 {amb}건 → {out}")

    # 최종 분리 재검증
    tu = {uid_of(r) for r in load("gen_main/c2_train_personas.jsonl")}
    hu = {it.get("uuid") for it in items if it.get("uuid")}
    print(f"재검증: H2 uuid {len(hu)}개, 학습셋과 교집합 {len(hu & tu)} (0이어야 함)")


if __name__ == "__main__":
    repair_generation()
    repair_h2()
