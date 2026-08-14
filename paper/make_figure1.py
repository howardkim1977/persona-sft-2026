# -*- coding: utf-8 -*-
"""
make_figure1.py — <Figure-01> 사용역 연령 격차의 단조 감소 (탐색적 발견)

학회 규격 대응: 2단 조판 1단 폭(75mm) 기준, 흑백 인쇄 가독성(회색조 + 마커
구분), 그림 제목은 본문에서 아래 가운데 배치하므로 이미지에는 제목을 넣지
않는다. 출력: figures/figure1_register_gap.png(600dpi) + .pdf
수치 출처: analysis/register_age_exploratory.txt (부트스트랩 10,000회)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# base → C0 → C1 → C2 (점추정, 95% CI)
labels = ["Base\n(no tuning)", "C0\n(no persona)", "C1\n(ad-hoc)",
          "C2\n(grounded)"]
point = [0.626, 0.275, 0.107, 0.019]
lo = [0.485, 0.157, -0.007, -0.115]
hi = [0.770, 0.390, 0.215, 0.155]
err_lo = [p - l for p, l in zip(point, lo)]
err_hi = [h - p for p, h in zip(point, hi)]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# 1단 폭 75mm ≈ 2.95in, 높이 비율 0.72
fig, ax = plt.subplots(figsize=(3.35, 2.45), dpi=600)

x = range(len(labels))
ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.errorbar(x, point, yerr=[err_lo, err_hi], fmt="none", ecolor="0.25",
            elinewidth=1.0, capsize=3, capthick=1.0, zorder=2)
ax.plot(x, point, color="0.25", lw=1.2, zorder=3)
# 유의(CI가 0 배제) = 채운 원, 비유의 = 빈 원
filled = [p_lo > 0 for p_lo in lo]
for xi, (p, f) in enumerate(zip(point, filled)):
    ax.plot(xi, p, marker="o", ms=5.5, mfc=("0.25" if f else "white"),
            mec="0.15", mew=1.0, zorder=4)

# 수치 라벨: 오차막대를 피해 좌우 교대 배치
for xi, (p, h) in enumerate(zip(point, hi)):
    dx, ha = (16, "left") if xi % 2 == 0 else (-16, "right")
    ax.annotate(f"{p:+.3f}", (xi, p), textcoords="offset points",
                xytext=(dx, -2), ha=ha, va="center", fontsize=7.2)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=7.2)
ax.set_ylabel("Register gap\n(under-40 minus 60+)", fontsize=7.8)
ax.set_ylim(-0.40, 0.88)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.tick_params(axis="both", labelsize=7.2, length=3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# 방향 안내 + 마커 범례 (작게)
ax.text(0.02, 0.985, "larger = worse for elderly speakers",
        transform=ax.transAxes, fontsize=6.6, color="0.35", va="top")
from matplotlib.lines import Line2D
leg = ax.legend(handles=[
    Line2D([], [], marker="o", ls="none", ms=5, mfc="0.25", mec="0.15",
           label="95% CI excludes 0"),
    Line2D([], [], marker="o", ls="none", ms=5, mfc="white", mec="0.15",
           label="CI includes 0")],
    loc="lower left", frameon=False, fontsize=6.6,
    handletextpad=0.4, borderpad=0.1, labelspacing=0.25)

fig.tight_layout(pad=0.4)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure1_register_gap.{ext}"),
                bbox_inches="tight", facecolor="white")
print("저장:", os.path.join(OUT, "figure1_register_gap.png"))
