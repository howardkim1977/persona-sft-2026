# -*- coding: utf-8 -*-
"""
similarity_check.py — 자체 유사도 점검 (학회 기준 15% 미만)

투고 원고 본문과 저자 선행 연구 2편(IEEE Access 투고본, ETRI J 투고본)의
어절 n-gram 중복률을 계산한다. 표절 검사 서비스(카피킬러 등)의 정확한 재현은
아니며, 자기표절 위험 구간을 사전에 찾아내기 위한 근사 점검이다.
기준: 6-gram 연속 일치(카피킬러 기본 유사 기준과 유사한 보수적 설정).
"""

import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.dirname(BASE)
N = 6


def clean_md(path):
    t = open(path, encoding="utf-8").read()
    # 메타/체크리스트/표지 블록 제외: 본문 Ⅰ~Ⅴ와 Abstract만
    start = t.find("## Abstract")
    end = t.find("## Reference")
    t = t[start:end] if start != -1 and end != -1 else t
    t = re.sub(r"\[Table-\d+\][^\n]*", " ", t)
    t = re.sub(r"\|[^\n]*\|", " ", t)          # 표 제거
    t = re.sub(r"[#*`\[\]()>_]", " ", t)
    return t


def clean_tex(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    t = re.sub(r"%.*", " ", t)
    t = re.sub(r"\\begin\{(table|figure|tabular|thebibliography)\}.*?"
               r"\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", t)
    t = re.sub(r"[{}$&\\~^_#]", " ", t)
    return t


def words(t):
    t = t.lower()
    t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
    return t.split()


def shingles(ws, n=N):
    return Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))


def overlap_rate(target_ws, source_shingles):
    """target의 n-gram 중 source에 존재하는 비율(어절 커버리지 기준)."""
    tsh = [tuple(target_ws[i:i + N]) for i in range(len(target_ws) - N + 1)]
    covered = [False] * len(target_ws)
    hits = 0
    for i, s in enumerate(tsh):
        if source_shingles.get(s):
            hits += 1
            for j in range(i, min(i + N, len(covered))):
                covered[j] = True
    return (hits / len(tsh) if tsh else 0.0,
            sum(covered) / len(covered) if covered else 0.0, tsh, covered)


def main():
    target_path = os.path.join(BASE, "paper/draft_ksaforum_en.md")
    tgt = words(clean_md(target_path))
    print(f"투고 원고 본문 어절 수: {len(tgt)}")

    sources = {
        "선행1 IEEE Access (persona-validation)":
            os.path.join(RESEARCH,
                         "persona-validation/IEEE_Access_Main_Manuscript/main.tex"),
        "선행2 ETRI J (ai-comparison)":
            os.path.join(RESEARCH, "ai-comparison/paper/manuscript_etrij.tex"),
    }

    all_sh = Counter()
    for name, p in sources.items():
        if not os.path.exists(p):
            print(f"  (없음) {name}: {p}")
            continue
        sws = words(clean_tex(p))
        sh = shingles(sws)
        all_sh.update(sh)
        rate, cov, _, _ = overlap_rate(tgt, sh)
        print(f"  {name}: 소스 {len(sws)}어절 → "
              f"{N}-gram 일치율 {rate*100:.2f}%, 어절 커버리지 {cov*100:.2f}%")

    rate, cov, tsh, covered = overlap_rate(tgt, all_sh)
    print(f"\n합산 (선행 연구 전체 대비): {N}-gram 일치율 {rate*100:.2f}%, "
          f"어절 커버리지 {cov*100:.2f}%")
    print(f"판정 기준(15%): {'통과' if cov < 0.15 else '주의 - 재작성 필요'}")

    # 중복 구간 상위 표시
    dup = [(i, s) for i, s in enumerate(tsh) if all_sh.get(s)]
    if dup:
        print(f"\n중복 구간 {len(dup)}건 중 최대 10건:")
        seen = set()
        shown = 0
        for i, s in dup:
            key = s[:3]
            if key in seen:
                continue
            seen.add(key)
            print("  -", " ".join(tgt[i:i + N + 4]))
            shown += 1
            if shown >= 10:
                break
    else:
        print("\n연속 일치 구간 없음")


if __name__ == "__main__":
    main()
