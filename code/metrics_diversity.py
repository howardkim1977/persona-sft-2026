# -*- coding: utf-8 -*-
"""
metrics_diversity.py — 파일럿 데이터 다양성/중복도 지표 예행 (Phase 0)

지표 (instruction 기준, response는 보조):
  - distinct-1/2/3      : 고유 n-gram 비율 (공백 토큰)
  - pairwise self-BLEU  : 쌍별 BLEU-4(스무딩) 평균. 높을수록 중복.
  - 압축비(zlib-9)      : 원문 바이트/압축 바이트. 높을수록 중복(Shaib 계열).
  - 임베딩 분산         : gemini-embedding-001(768차원), 쌍별 코사인 거리 평균
                          + 중심 거리 평균. 높을수록 다양.
사전등록 시 이 스위트를 H1 지표로 고정한다.
"""

import json
import math
import os
import zlib
import collections
import time
from urllib import request as urlreq

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_GEMINI = os.path.expanduser("~/Research/persona-validation/.env")
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768


def load_gemini_key() -> str:
    with open(ENV_GEMINI) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("GEMINI_API_KEY 없음")


def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


def distinct_n(texts, n):
    all_ngrams = set()
    total = 0
    for t in texts:
        gs = ngrams(t.split(), n)
        all_ngrams.update(gs)
        total += len(gs)
    return len(all_ngrams) / total if total else 0.0


def sentence_bleu(cand_tokens, ref_tokens, max_n=4):
    """스무딩(+1) BLEU-4, 단일 참조."""
    if not cand_tokens or not ref_tokens:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        c_ngr = collections.Counter(ngrams(cand_tokens, n))
        r_ngr = collections.Counter(ngrams(ref_tokens, n))
        overlap = sum(min(c, r_ngr[g]) for g, c in c_ngr.items())
        total = max(sum(c_ngr.values()), 1)
        log_p += math.log((overlap + 1) / (total + 1))
    log_p /= max_n
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens))) \
        if len(cand_tokens) < len(ref_tokens) else 1.0
    # 표준 BP는 cand<ref 시 패널티. 위 구현: cand가 짧으면 exp(1-ref/cand)<1.
    bp = math.exp(1 - len(ref_tokens) / len(cand_tokens)) \
        if len(cand_tokens) < len(ref_tokens) else 1.0
    return bp * math.exp(log_p)


def pairwise_self_bleu(texts):
    toks = [t.split() for t in texts]
    n = len(toks)
    s = 0.0
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += sentence_bleu(toks[i], toks[j])
                cnt += 1
    return s / cnt if cnt else 0.0


def compression_ratio(texts):
    blob = "\n".join(texts).encode("utf-8")
    comp = zlib.compress(blob, 9)
    return len(blob) / len(comp)


def embed_batch(key, texts):
    reqs = [{"model": EMBED_MODEL,
             "content": {"parts": [{"text": t}]},
             "outputDimensionality": EMBED_DIM} for t in texts]
    body = json.dumps({"requests": reqs}).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"{EMBED_MODEL}:batchEmbedContents?key={key}")
    for attempt in range(4):
        try:
            req = urlreq.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
            with urlreq.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return [e["values"] for e in data["embeddings"]]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 3)


def embedding_dispersion(key, texts, batch=100):
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.extend(embed_batch(key, texts[i:i + batch]))
    X = np.array(vecs, dtype=np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = X @ X.T
    n = len(X)
    iu = np.triu_indices(n, k=1)
    mean_pair_dist = float(1 - sims[iu].mean())
    centroid = X.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    mean_centroid_dist = float(1 - (X @ centroid).mean())
    return mean_pair_dist, mean_centroid_dist


def main():
    rows = [json.loads(l) for l in open(os.path.join(BASE, "pilot/pilot_pairs.jsonl"))]
    key = load_gemini_key()
    report = {}
    for cond in ("C0", "C1", "C2"):
        ins = [r["pair"]["instruction"] for r in rows
               if r["condition"] == cond and r["ok"]]
        res = [r["pair"]["response"] for r in rows
               if r["condition"] == cond and r["ok"]]
        print(f"[{cond}] n={len(ins)} 계산 중...")
        pd, cd = embedding_dispersion(key, ins)
        report[cond] = {
            "n": len(ins),
            "distinct1": distinct_n(ins, 1),
            "distinct2": distinct_n(ins, 2),
            "distinct3": distinct_n(ins, 3),
            "self_bleu": pairwise_self_bleu(ins),
            "compression_ratio_instr": compression_ratio(ins),
            "compression_ratio_resp": compression_ratio(res),
            "embed_pairwise_dist": pd,
            "embed_centroid_dist": cd,
        }
    out = os.path.join(BASE, "pilot/diversity_report.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)

    print("\n=== 다양성 지표 (instruction, 높을수록 다양: distinct/embed, "
          "낮을수록 다양: self-BLEU/압축비) ===")
    keys = ["distinct1", "distinct2", "distinct3", "self_bleu",
            "compression_ratio_instr", "compression_ratio_resp",
            "embed_pairwise_dist", "embed_centroid_dist"]
    header = "지표".ljust(26) + "".join(c.rjust(9) for c in ("C0", "C1", "C2"))
    print(header)
    for k in keys:
        line = k.ljust(26)
        for cond in ("C0", "C1", "C2"):
            line += f"{report[cond][k]:9.4f}"
        print(line)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
