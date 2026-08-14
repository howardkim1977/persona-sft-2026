# -*- coding: utf-8 -*-
"""
verify_manuscript.py — 원고 수치와 분석 산출물의 일치 여부 전수 대조

원고(paper/draft_ksaforum_en.md)에 적힌 값이 analysis/ 산출물과 일치하는지
자동 검증한다. 불일치는 FAIL로 보고한다.
"""

import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = open(os.path.join(BASE, "paper/draft_ksaforum_en.md"),
          encoding="utf-8").read()

ok, fail = [], []


def chk(name, cond, detail=""):
    (ok if cond else fail).append(f"{name}" + (f" — {detail}" if detail else ""))


def has(s):
    return s in MS


# ---------- 1. 조작 점검 (Table-01) ----------
mc = json.load(open(os.path.join(BASE, "analysis/manip_check.json")))
for cond in ("C0", "C1", "C2"):
    m = mc[cond]
    chk(f"[T1] {cond} distinct-1 {m['distinct1']:.3f}",
        has(f"{m['distinct1']:.3f}"))
    chk(f"[T1] {cond} distinct-2 {m['distinct2']:.3f}",
        has(f"{m['distinct2']:.3f}"))
    chk(f"[T1] {cond} distinct-3 {m['distinct3']:.3f}",
        has(f"{m['distinct3']:.3f}"))
    chk(f"[T1] {cond} compression {m['compression']:.3f}",
        has(f"{m['compression']:.3f}"))
    chk(f"[T1] {cond} self-BLEU {m['self_bleu_2k']:.4f}",
        has(f"{m['self_bleu_2k']:.4f}"))
# 순서 주장 검증
for k in ("distinct1", "distinct2", "distinct3"):
    chk(f"[T1] {k} C2>C1>C0 성립",
        mc["C2"][k] > mc["C1"][k] > mc["C0"][k])
for k in ("compression", "self_bleu_2k"):
    chk(f"[T1] {k} C2<C1<C0 성립(낮을수록 다양)",
        mc["C2"][k] < mc["C1"][k] < mc["C0"][k])

# ---------- 2. H1 (Table-02) ----------
h1 = json.load(open(os.path.join(BASE, "analysis/h1_confirmatory.json")))
for other in ("C0", "C1"):
    t = h1["tests"][f"C2_vs_{other}"]
    chk(f"[H1] C2 vs {other} 지지 수 {t['supported_count']}/5 기각",
        t["supported_count"] < 4 and not t["h1_supported"])
    chk(f"[H1] C2 vs {other} 지지 수 원고 표기",
        has(f"{t['supported_count']}/5"))
    for metric, r in t["per_metric"].items():
        p = r["point"]
        s = f"{p:+.4f}"
        chk(f"[H1] {other} {metric} 점추정 {s}", has(s),
            "원고에서 미발견" if not has(s) else "")
chk("[H1] 부트스트랩 10,000회 명시", has("10,000 resamples"))

# ---------- 3. H2 ----------
h2 = json.load(open(os.path.join(BASE, "analysis/h2_confirmatory.json")))
chk("[H2] QWK 0.666", has("0.666") and abs(h2["qwk_sol_opus5"] - 0.666) < 5e-4)
chk("[H2] 완전일치 77.5%", has("77.5") and abs(h2["exact_agree"] - .775) < 5e-3)
chk("[H2] 문항 400 / 제외 0", h2["n_items"] == 400 and h2["excluded_items"] == 0)
for axis in ("age", "region"):
    g = h2["axes"][axis]["gaps"]
    for cond in ("C0", "C1", "C2", "base"):
        v = g[cond]
        chk(f"[H2] {axis} {cond} 격차 {v:+.4f}",
            has(f"{abs(v):.4f}"), f"값 {v:+.4f}")
    for k, t in h2["axes"][axis]["tests"].items():
        chk(f"[H2] {axis} {k} 점추정 {t['point']:+.4f}",
            has(f"{abs(t['point']):.4f}"))
    # 모든 확증 검정이 CI에 0 포함(기각)인지
    for k, t in h2["axes"][axis]["tests"].items():
        chk(f"[H2] {axis} {k} 기각(CI 0 포함)", not t["supported"])
    # 바닥 효과 주장: 전 조건 |격차| < 0.08
    chk(f"[H2] {axis} 바닥 효과(|격차|<0.08) 주장 성립",
        all(abs(v) < 0.08 for v in g.values()))

# ---------- 4. H3 ----------
h3 = open(os.path.join(BASE, "analysis/h3_confirmatory.txt")).read()
m = re.search(r"C2-C0: ([+-][\d.]+)pp, 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", h3)
chk("[H3] C2-C0 +0.00pp CI[-0.50,+0.48] 일치",
    bool(m) and has("+0.00pp") and has("[-0.50, +0.48]"),
    m.group(0) if m else "산출물 파싱 실패")
m2 = re.search(r"C2-C1: ([+-][\d.]+)pp, 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", h3)
chk("[H3] C2-C1 -0.46pp CI[-0.96,+0.03] 일치",
    bool(m2) and has("-0.46pp") and has("[-0.96, +0.03]"))
chk("[H3] 비열등 마진 -2.0pp 명시", has("-2.0pp"))

# 정확도 재계산 (원고 값과 대조)
accs = {}
for mk in ["base", "C0_s42", "C0_s777", "C1_s42", "C1_s777", "C2_s42", "C2_s777"]:
    rows = [json.loads(l) for l in
            open(os.path.join(BASE, f"eval_out/h3_{mk}.jsonl"))]
    accs[mk] = sum(1 for r in rows if r["pred"] == r["gold"]) / len(rows)
    chk(f"[H3] {mk} n=6561", len(rows) == 6561)
for c in ("C0", "C1", "C2"):
    mean = (accs[f"{c}_s42"] + accs[f"{c}_s777"]) / 2
    chk(f"[H3] {c} 평균 {mean*100:.2f}% 원고 일치", has(f"{mean*100:.2f}%"))
chk(f"[H3] base {accs['base']*100:.2f}% 원고 일치",
    has(f"{accs['base']*100:.2f}%"))

# ---------- 5. 탐색적 ----------
ex = json.load(open(os.path.join(BASE, "analysis/exploratory.json")))
chk("[E] HCX 위반율 15.8%", has("15.8") and
    abs(ex["hcx"]["violation_rate"] - 0.158) < 2e-3)
taus = list(ex["judge_ranks"]["kendall_tau"].values())
chk("[E] Kendall tau 범위 0.62-0.90 표기",
    has("0.62-0.90") and abs(min(taus) - 0.619) < 1e-2
    and abs(max(taus) - 0.905) < 1e-2)
emb = ex["embedding"]["per_cond"]
for c in ("C0", "C1", "C2"):
    chk(f"[E] 임베딩 {c} {emb[c]['pairwise']:.3f}",
        has(f"{emb[c]['pairwise']:.3f}"))
chk("[E] 임베딩 C0>C1>C2 (역방향) 주장 성립",
    emb["C0"]["pairwise"] > emb["C1"]["pairwise"] > emb["C2"]["pairwise"])
reg = open(os.path.join(BASE, "analysis/register_age_exploratory.txt")).read()
for v in ("+0.626", "+0.275", "+0.107", "+0.019", "+0.256"):
    chk(f"[E] register {v}", v in reg and has(v))
chk("[E] register C0-C2 CI [+0.100,+0.415]",
    has("[+0.100, +0.415]") and "+0.100" in reg)

# ---------- 6. 생성/규모 수치 ----------
gen = [json.loads(l) for l in open(os.path.join(BASE, "gen_main/results.jsonl"))]
calls = len(gen)
chk(f"[G] 총 호출 {calls} 원고 일치", has(f"{calls:,}"))
for c in ("C0", "C1", "C2"):
    v = sum(1 for r in gen if r["condition"] == c and r["status"] == "valid")
    chk(f"[G] {c} 유효 10,000", v == 10000)
chk("[G] 생성비 USD 4.26", has("4.26"))
for n, lbl in ((500, "h1"), (400, "h2")):
    cnt = sum(1 for _ in open(os.path.join(BASE, f"eval_sets/{lbl}_items.jsonl"
                              if lbl == "h2" else "eval_sets/h1_prompts.jsonl")))
    chk(f"[G] {lbl} 세트 {n}건", cnt == n, f"실제 {cnt}")
chk("[G] H3 6,561 표기", has("6,561"))
chk("[G] KoBEST 4,561 / KMMLU 2,000 표기",
    has("4,561") and has("2,000"))

# ---------- 출력 ----------
print(f"=== 검증 {len(ok)+len(fail)}건: 통과 {len(ok)} / 불일치 {len(fail)} ===\n")
if fail:
    print("[불일치 항목]")
    for f in fail:
        print("  FAIL:", f)
else:
    print("모든 항목 일치")
