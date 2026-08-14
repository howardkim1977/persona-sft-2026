# -*- coding: utf-8 -*-
"""
manip_check.py — 조작 점검: 본 생성 데이터(조건당 10,000 instruction)의
데이터 수준 어휘 다양성 (사전등록 H1의 manipulation check 항목)

distinct-1/2/3, 압축비: 전체 10,000. 쌍별 self-BLEU: 시드 고정 표본 2,000
(계산 제약, 결과에 명기). 파일럿(C2>C1>C0) 재현 여부 확인.
"""

import json
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from metrics_diversity import distinct_n, pairwise_self_bleu, compression_ratio

rows = [json.loads(l) for l in open(os.path.join(BASE, "gen_main/results.jsonl"))]
out = {"self_bleu_subsample": 2000}
for cond in ("C0", "C1", "C2"):
    ins = [r["pair"]["instruction"] for r in rows
           if r["condition"] == cond and r["status"] == "valid"]
    ins.sort()
    rng = random.Random(42)
    sub = rng.sample(ins, 2000)
    m = {
        "n": len(ins),
        "distinct1": distinct_n(ins, 1),
        "distinct2": distinct_n(ins, 2),
        "distinct3": distinct_n(ins, 3),
        "compression": compression_ratio(ins),
        "self_bleu_2k": pairwise_self_bleu(sub),
    }
    out[cond] = {k: round(v, 5) if isinstance(v, float) else v
                 for k, v in m.items()}
    print(cond, out[cond], flush=True)

json.dump(out, open(os.path.join(BASE, "analysis/manip_check.json"), "w"),
          ensure_ascii=False, indent=2)
print("저장: analysis/manip_check.json")
