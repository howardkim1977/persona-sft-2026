# -*- coding: utf-8 -*-
"""
h2_analysis.py — 주가설 H2 확증 분석 (사전등록 v0.2 6.2절)

주지표: 조건별 내용 적합성 격차 = (기준집단 평균 - 하위집단 평균).
  기준집단: 연령 축 under40, 지역 축 metro.
지지 기준: C2 격차 < C1 격차 및 C2 격차 < C0 격차, 각각 층화(집단별) 문항
부트스트랩 10,000회 95% CI가 0을 배제.
점수: 주심판 2인(sol, opus5) 평균, 조건 = 시드 2 runs 평균.
부가: 심판 간 일치도(QWK), base 참고치, register(탐색적) 격차.
"""

import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = ["s42", "s777"]
CONDS = ["C0", "C1", "C2"]
REF = {"age": "under40", "region": "metro"}
SUB = {"age": "60plus", "region": "nonmetro"}
N_BOOT = 10000


def qwk(a, b, k=4):
    a = np.asarray(a, int); b = np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    ha = O.sum(1); hb = O.sum(0)
    E = np.outer(ha, hb) / len(a)
    return 1 - (w * O).sum() / (w * E).sum()


def main():
    rows = [json.loads(l) for l in open(os.path.join(BASE, "judge_out/scores.jsonl"))]
    prim = [r for r in rows if r["judge"] in ("sol", "opus5") and r["score"]]

    # (model, item) -> {judge: content}
    from collections import defaultdict
    sc = defaultdict(dict)
    meta = {}
    for r in prim:
        sc[(r["model"], r["item"])][r["judge"]] = r["score"]["content"]
        meta[r["item"]] = (r["axis"], r["group"])

    # 심판 일치도 (전 응답)
    pairs = [(v["sol"], v["opus5"]) for v in sc.values()
             if "sol" in v and "opus5" in v]
    q = qwk([p[0] for p in pairs], [p[1] for p in pairs])
    exact = np.mean([p[0] == p[1] for p in pairs])

    # 문항 x 조건 점수 행렬 (주심판 평균 -> 시드 평균)
    items = sorted(meta.keys(), key=lambda k: int(k.split(":")[1]))
    def cond_score(cond, item):
        vals = []
        for s in SEEDS:
            v = sc.get((f"{cond}_{s}", item))
            if v and "sol" in v and "opus5" in v:
                vals.append((v["sol"] + v["opus5"]) / 2)
        return np.mean(vals) if len(vals) == 2 else None

    # 결측(주심판 실패) 문항 전 모델 제외 규칙
    valid_items = []
    for it in items:
        if all(cond_score(c, it) is not None for c in CONDS) \
                and sc.get(("base", it)) and len(sc[("base", it)]) == 2:
            valid_items.append(it)
    excluded = len(items) - len(valid_items)

    S = {c: {} for c in CONDS + ["base"]}
    for it in valid_items:
        for c in CONDS:
            S[c][it] = cond_score(c, it)
        v = sc[("base", it)]
        S["base"][it] = (v["sol"] + v["opus5"]) / 2

    out = {"qwk_sol_opus5": round(float(q), 4),
           "exact_agree": round(float(exact), 4),
           "n_items": len(valid_items), "excluded_items": excluded,
           "axes": {}}
    rng = np.random.default_rng(42)
    print(f"주심판 QWK {q:.3f}, 완전일치 {exact:.3f}, "
          f"유효 문항 {len(valid_items)} (제외 {excluded})")

    for axis in ("age", "region"):
        ref_items = [it for it in valid_items if meta[it] == (axis, REF[axis])]
        sub_items = [it for it in valid_items if meta[it] == (axis, SUB[axis])]
        nr, ns = len(ref_items), len(sub_items)
        gaps = {}
        for c in CONDS + ["base"]:
            gaps[c] = float(np.mean([S[c][i] for i in ref_items])
                            - np.mean([S[c][i] for i in sub_items]))
        # 층화 부트스트랩: 집단별 문항 재추출, 조건 간은 동일 문항으로 대응
        ref_mat = {c: np.array([S[c][i] for i in ref_items]) for c in CONDS}
        sub_mat = {c: np.array([S[c][i] for i in sub_items]) for c in CONDS}
        ridx = rng.integers(0, nr, (N_BOOT, nr))
        sidx = rng.integers(0, ns, (N_BOOT, ns))
        bg = {c: ref_mat[c][ridx].mean(1) - sub_mat[c][sidx].mean(1)
              for c in CONDS}
        res = {"gaps": {c: round(gaps[c], 4) for c in gaps},
               "n_ref": nr, "n_sub": ns, "tests": {}}
        print(f"\n[{axis} 축] 격차(기준-하위): " +
              ", ".join(f"{c} {gaps[c]:+.3f}" for c in CONDS + ["base"]))
        for other in ("C1", "C0"):
            diff = bg[other] - bg["C2"]        # >0 이면 C2 격차가 더 작음
            point = gaps[other] - gaps["C2"]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            support = bool(point > 0 and lo > 0)
            res["tests"][f"{other}-C2"] = {
                "point": round(float(point), 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
                "supported": support}
            print(f"  격차 차이 {other}-C2: {point:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] → "
                  f"{'지지' if support else '기각'}")
        out["axes"][axis] = res

    json.dump(out, open(os.path.join(BASE, "analysis/h2_confirmatory.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n저장: analysis/h2_confirmatory.json")


if __name__ == "__main__":
    main()
