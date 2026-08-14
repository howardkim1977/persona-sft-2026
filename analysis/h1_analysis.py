# -*- coding: utf-8 -*-
"""
h1_analysis.py — H1 확증 분석 (사전등록 v0.2 6.1절)

지표 5종(distinct-1/2/3, 쌍별 self-BLEU, zlib-9 압축비)을 모델 응답에 계산.
조건 = 시드 2 runs 평균. 조건 간 차이는 프롬프트 대응 부트스트랩 10,000회 95% CI.
지지 기준: C2 vs C0, C2 vs C1 각각에서 5지표 중 4개 이상이 다양성 우위 방향으로
CI가 0을 배제.

주의: distinct-n/self-BLEU/압축비는 집합 수준 지표이므로, 부트스트랩은 프롬프트
인덱스를 재추출해 응답 집합을 재구성한 뒤 지표를 재계산한다(조건 간 동일 인덱스
= 프롬프트 대응). self-BLEU는 모델별 500x500 쌍별 BLEU 행렬을 사전 계산한 뒤
부트스트랩에서 행렬 인덱싱으로 평균만 재계산한다(전 지표 10,000회 유지).
"""

import json
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from metrics_diversity import distinct_n, pairwise_self_bleu, compression_ratio

SEEDS = ["s42", "s777"]
CONDS = ["C0", "C1", "C2"]
N_BOOT = 10000


def bleu_matrix(texts):
    """쌍별 BLEU 행렬 (대각 0). 부트스트랩용 사전 계산."""
    from metrics_diversity import sentence_bleu
    toks = [t.split() for t in texts]
    n = len(toks)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = sentence_bleu(toks[i], toks[j])
    return M


def load_responses(model):
    p = os.path.join(BASE, f"eval_out/h1_{model}.jsonl")
    rows = [json.loads(l) for l in open(p)]
    rows.sort(key=lambda r: int(r["key"].split(":")[1]))
    return [r["response"] for r in rows]


def metric_on(texts, name):
    if name == "distinct1":
        return distinct_n(texts, 1)
    if name == "distinct2":
        return distinct_n(texts, 2)
    if name == "distinct3":
        return distinct_n(texts, 3)
    if name == "self_bleu":
        return pairwise_self_bleu(texts)
    return compression_ratio(texts)


# 다양성 우위 방향: distinct는 +, self_bleu/압축비는 -
DIRECTION = {"distinct1": 1, "distinct2": 1, "distinct3": 1,
             "self_bleu": -1, "compression": -1}


def main():
    resp = {}
    for c in CONDS:
        for s in SEEDS:
            resp[f"{c}_{s}"] = load_responses(f"{c}_{s}")
    resp["base"] = load_responses("base")
    n = len(resp["C0_s42"])
    rng = np.random.default_rng(42)

    out = {"n_prompts": n, "metrics": {}, "tests": {}, "n_boot": N_BOOT}

    # 점추정 (시드 평균)
    for name in DIRECTION:
        vals = {}
        for c in CONDS:
            vals[c] = np.mean([metric_on(resp[f"{c}_{s}"], name) for s in SEEDS])
        vals["base"] = metric_on(resp["base"], name)
        out["metrics"][name] = {k: round(float(v), 5) for k, v in vals.items()}
        print(f"{name}: " + ", ".join(f"{k} {v:.4f}" for k, v in vals.items()))

    # --- 사전 계산 (6개 조정 모델) ---
    tuned = [f"{c}_{s}" for c in CONDS for s in SEEDS]
    import zlib
    from metrics_diversity import ngrams, sentence_bleu

    print("사전 계산: BLEU 행렬·n-gram id·바이트열 (6모델)...", flush=True)
    M_bleu, NG, RAW = {}, {}, {}
    for mkey in tuned:
        texts = resp[mkey]
        toks = [t.split() for t in texts]
        Mx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Mx[i, j] = sentence_bleu(toks[i], toks[j]) if i != j else \
                    sentence_bleu(toks[i], toks[i])
        M_bleu[mkey] = Mx
        # n-gram 정수 id 배열 (distinct 고속화)
        NG[mkey] = {}
        for nn in (1, 2, 3):
            vocab = {}
            arrs = []
            for t in toks:
                ids = []
                for g in ngrams(t, nn):
                    if g not in vocab:
                        vocab[g] = len(vocab)
                    ids.append(vocab[g])
                arrs.append(np.array(ids, dtype=np.int64))
            NG[mkey][nn] = arrs
        RAW[mkey] = [t.encode("utf-8") for t in texts]
        print(f"  {mkey} 완료", flush=True)

    def boot_metrics(mkey, idx):
        r = {}
        S = M_bleu[mkey][np.ix_(idx, idx)]
        r["self_bleu"] = (S.sum() - np.trace(S)) / (n * (n - 1))
        for nn, key in ((1, "distinct1"), (2, "distinct2"), (3, "distinct3")):
            cat = np.concatenate([NG[mkey][nn][i] for i in idx])
            r[key] = len(np.unique(cat)) / len(cat) if len(cat) else 0.0
        blob = b"\n".join(RAW[mkey][i] for i in idx)
        r["compression"] = len(blob) / len(zlib.compress(blob, 9))
        return r

    print(f"부트스트랩 {N_BOOT}회 (프롬프트 대응, 전 지표)...", flush=True)
    diffs = {f"C2_vs_{o}": {k: np.empty(N_BOOT) for k in DIRECTION}
             for o in ("C0", "C1")}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals = {mkey: boot_metrics(mkey, idx) for mkey in tuned}
        for name in DIRECTION:
            c2 = (vals["C2_s42"][name] + vals["C2_s777"][name]) / 2
            for o in ("C0", "C1"):
                vo = (vals[f"{o}_s42"][name] + vals[f"{o}_s777"][name]) / 2
                diffs[f"C2_vs_{o}"][name][b] = (c2 - vo) * DIRECTION[name]
        if (b + 1) % 1000 == 0:
            print(f"  {b+1}/{N_BOOT}", flush=True)

    for o in ("C0", "C1"):
        support_cnt = 0
        tests = {}
        for name in DIRECTION:
            d = diffs[f"C2_vs_{o}"][name]
            lo, hi = np.percentile(d, [2.5, 97.5])
            point = float(np.mean(d))
            sup = bool(lo > 0)
            support_cnt += sup
            tests[name] = {"point": round(point, 5),
                           "ci": [round(float(lo), 5), round(float(hi), 5)],
                           "supported": sup}
            print(f"  C2 vs {o} {name}: {point:+.5f} "
                  f"[{lo:+.5f}, {hi:+.5f}] {'지지' if sup else '기각'}")
        overall = support_cnt >= 4
        out["tests"][f"C2_vs_{o}"] = {
            "per_metric": tests, "supported_count": support_cnt,
            "h1_supported": overall}
        print(f"C2 vs {o}: {support_cnt}/5 지지 → H1 "
              f"{'지지' if overall else '기각'}\n")

    json.dump(out, open(os.path.join(BASE, "analysis/h1_confirmatory.json"), "w"),
              ensure_ascii=False, indent=2)
    print("저장: analysis/h1_confirmatory.json")


if __name__ == "__main__":
    main()
