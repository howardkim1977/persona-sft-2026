# -*- coding: utf-8 -*-
"""
exploratory.py — 사전등록에 명시된 탐색적 분석 일괄

(a) 임베딩 분산: H1 응답 500개/모델, gemini-embedding-001 768차원.
    조건 = 시드 평균. 쌍별 코사인 거리·중심 거리.
(b) register 격차: H2 register 점수(주심판 평균)로 내용 적합성과 동일한
    격차 분석 (탐색적).
(c) 4심판 순위 안정성: 심판별 7모델 순위 + Kendall tau.
(d) HCX 척도 위반: 0-3 이탈률, 4→3 절사 민감도에서의 주심판과의 상관.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from metrics_diversity import load_gemini_key, embedding_dispersion

MODELS = ["base", "C0_s42", "C0_s777", "C1_s42", "C1_s777", "C2_s42", "C2_s777"]
CONDS = ["C0", "C1", "C2"]
SEEDS = ["s42", "s777"]
out = {}

# ---------- (a) 임베딩 분산 ----------
print("(a) 임베딩 분산 계산 중 (3,500 임베딩)...", flush=True)
key = load_gemini_key()
emb = {}
for m in MODELS:
    rows = [json.loads(l) for l in open(os.path.join(BASE, f"eval_out/h1_{m}.jsonl"))]
    rows.sort(key=lambda r: int(r["key"].split(":")[1]))
    texts = [r["response"][:4000] for r in rows]
    pd_, cd_ = embedding_dispersion(key, texts)
    emb[m] = {"pairwise": round(pd_, 5), "centroid": round(cd_, 5)}
    print(f"  {m}: pairwise {pd_:.4f} centroid {cd_:.4f}", flush=True)
out["embedding"] = {
    "per_model": emb,
    "per_cond": {c: {k: round(np.mean([emb[f'{c}_{s}'][k] for s in SEEDS]), 5)
                     for k in ("pairwise", "centroid")} for c in CONDS},
}

# ---------- (b)(c)(d) 심판 데이터 ----------
rows = [json.loads(l) for l in open(os.path.join(BASE, "judge_out/scores.jsonl"))]
meta = {}
sc = defaultdict(dict)     # (model,item) -> judge -> (content, register)
raw_fail = defaultdict(int)
hcx_viol = 0
hcx_total = 0
for r in rows:
    meta[r["item"]] = (r["axis"], r["group"])
    if r["judge"] == "hcx":
        hcx_total += 1
        if not r["score"]:
            raw = r.get("raw") or ""
            if '"content"' in raw:
                hcx_viol += 1
    if r["score"]:
        sc[(r["model"], r["item"])][r["judge"]] = (
            r["score"]["content"], r["score"]["register"])

items = sorted(meta.keys(), key=lambda k: int(k.split(":")[1]))

# (b) register 격차 (주심판 평균, 시드 평균)
REF = {"age": "under40", "region": "metro"}
SUB = {"age": "60plus", "region": "nonmetro"}
def reg_score(model, item):
    v = sc.get((model, item))
    if v and "sol" in v and "opus5" in v:
        return (v["sol"][1] + v["opus5"][1]) / 2
    return None

reg = {"axes": {}}
for axis in ("age", "region"):
    res = {}
    for cond in CONDS + ["base"]:
        def cscore(item):
            if cond == "base":
                return reg_score("base", item)
            vals = [reg_score(f"{cond}_{s}", item) for s in SEEDS]
            return None if any(v is None for v in vals) else float(np.mean(vals))
        r_ = [cscore(i) for i in items if meta[i] == (axis, REF[axis])]
        s_ = [cscore(i) for i in items if meta[i] == (axis, SUB[axis])]
        r_ = [v for v in r_ if v is not None]
        s_ = [v for v in s_ if v is not None]
        res[cond] = round(float(np.mean(r_) - np.mean(s_)), 4)
    reg["axes"][axis] = res
    print(f"(b) register 격차 [{axis}]: {res}", flush=True)
out["register_gaps"] = reg

# (c) 심판별 모델 순위 + Kendall tau
from itertools import combinations
def kendall(a, b):
    n = len(a)
    conc = disc = 0
    for i, j in combinations(range(n), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else 0.0

judge_means = {}
for jkey in ("sol", "opus5", "gpt52", "hcx"):
    means = []
    for m in MODELS:
        vals = [sc[(m, i)][jkey][0] for i in items
                if jkey in sc.get((m, i), {})]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    judge_means[jkey] = means
    order = sorted(range(len(MODELS)), key=lambda k: -means[k])
    print(f"(c) {jkey} 순위: " +
          " > ".join(f"{MODELS[k]}({means[k]:.2f})" for k in order), flush=True)
taus = {}
for a, b in combinations(("sol", "opus5", "gpt52", "hcx"), 2):
    va, vb = judge_means[a], judge_means[b]
    mask = [i for i in range(len(MODELS))
            if not (np.isnan(va[i]) or np.isnan(vb[i]))]
    taus[f"{a}-{b}"] = round(kendall([va[i] for i in mask],
                                     [vb[i] for i in mask]), 3)
out["judge_ranks"] = {"means": {k: [round(x, 4) for x in v]
                               for k, v in judge_means.items()},
                      "models": MODELS, "kendall_tau": taus}
print("(c) Kendall tau:", taus, flush=True)

# (d) HCX 척도 위반
out["hcx"] = {"total": hcx_total, "scale_violation": hcx_viol,
              "violation_rate": round(hcx_viol / hcx_total, 4)}
print(f"(d) HCX 척도 위반: {hcx_viol}/{hcx_total} "
      f"({hcx_viol/hcx_total*100:.1f}%)", flush=True)

json.dump(out, open(os.path.join(BASE, "analysis/exploratory.json"), "w"),
          ensure_ascii=False, indent=2)
print("저장: analysis/exploratory.json")
