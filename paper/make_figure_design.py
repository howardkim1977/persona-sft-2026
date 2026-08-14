# -*- coding: utf-8 -*-
"""
make_figure_design.py — <Figure-01> 연구 설계 개관도

2단 조판 1단 폭(75mm), 흑백 인쇄 대응. 그림 제목은 본문에서 아래 가운데
배치하므로 이미지에 넣지 않는다.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.4})
fig, ax = plt.subplots(figsize=(3.35, 3.15), dpi=600)
ax.set_xlim(0, 10)
ax.set_ylim(0, 11.4)
ax.axis("off")


def box(x, y, w, h, text, fill="white", lw=0.8, fs=6.4, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.10,rounding_size=0.12",
                                fc=fill, ec="0.15", lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), lw=lw, color="0.25",
                                 ls=ls, arrowstyle="-|>", mutation_scale=7,
                                 shrinkA=1, shrinkB=1, zorder=1))


# 공통 입력
box(0.1, 10.2, 9.8, 0.85,
    "8 task types, identical list and allocation across conditions", "0.92")

# 조건 3개 (페르소나 출처는 상자 안에 표기 — 화살표 교차 제거)
ys = 8.25
box(0.1, ys, 3.0, 1.45, "C0\nno persona\n(task only)", "white", fs=6.0)
box(3.5, ys, 3.0, 1.45, "C1\nad-hoc persona\n(random fields)",
    "white", fs=6.0)
box(6.9, ys, 3.0, 1.45, "C2\ngrounded persona\n(Nemotron-Korea)",
    "0.85", lw=1.4, bold=True, fs=6.0)
for x in (1.6, 5.0, 8.4):
    arrow(x, 10.2, x, 9.70)

# 생성
box(0.1, 7.0, 9.8, 0.95,
    "Generation: DeepSeek-V4-Flash-0731 (MIT), temp 1.0\n"
    "10,000 valid pairs per condition, identical filter", "white")
for x in (1.6, 5.0, 8.4):
    arrow(x, ys, x, 7.95)

# 학습
box(0.1, 5.65, 9.8, 0.95,
    "LoRA fine-tuning: EXAONE-3.5-2.4B-Instruct\n"
    "2 seeds per condition -> 6 tuned models + base", "white")
arrow(5.0, 7.0, 5.0, 6.60)

# 평가 3축
box(0.1, 3.9, 3.0, 1.35, "H1\nlexical diversity\n500 neutral prompts\n"
                         "5 metrics", "white", fs=6.0)
box(3.5, 3.9, 3.0, 1.35, "H2 (primary)\nsubgroup gap\n400 items\n"
                         "4 judges", "0.85", lw=1.4, fs=6.0)
box(6.9, 3.9, 3.0, 1.35, "H3\nnon-inferiority\n6,561 items\n"
                         "KoBEST + KMMLU", "white", fs=6.0)
for x in (1.6, 5.0, 8.4):
    arrow(x, 5.65, x, 5.25)

# 판정
box(0.1, 2.5, 3.0, 0.95, "rejected\n1-2 of 5 metrics", "white")
box(3.5, 2.5, 3.0, 0.95, "rejected\nfloor effect", "white")
box(6.9, 2.5, 3.0, 0.95, "supported\nCI > -2.0pp", "white")
for x in (1.6, 5.0, 8.4):
    arrow(x, 3.9, x, 3.45)

# 탐색적
box(1.5, 1.0, 7.0, 1.05,
    "Exploratory: register age gap collapses\n"
    "base +0.626 -> C2 +0.019 (Figure-02)", "white", ls="--")
arrow(5.0, 2.5, 5.0, 2.05, ls="--")

ax.text(0.1, 0.35, "Preregistered before data collection (OSF fc8mn)",
        fontsize=6.0, color="0.35")

fig.tight_layout(pad=0.2)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure_design.{ext}"),
                bbox_inches="tight", facecolor="white")
print("저장:", os.path.join(OUT, "figure_design.png"))
