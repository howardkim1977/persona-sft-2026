#!/bin/bash
# 전체 평가 실행: 7 모델 x (h1, h2, h3). 학습 완료 후 실행할 것.
# h1/h2 먼저(심판 채점의 선행물), h3는 마지막.
cd "$(dirname "$0")"
for part in h2 h1 h3; do
  for m in base C0_s42 C0_s777 C1_s42 C1_s777 C2_s42 C2_s777; do
    echo "[eval] $part $m $(date '+%H:%M:%S')"
    python3 eval_local.py --model "$m" --part "$part"
  done
done
echo "ALL EVAL DONE"
