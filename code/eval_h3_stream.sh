#!/bin/bash
# h3 전용 병렬 스트림 (본 드라이버의 h2/h1과 동시 실행, 재개 로직으로 충돌 없음)
cd "$(dirname "$0")"
for m in base C0_s42 C0_s777 C1_s42 C1_s777 C2_s42 C2_s777; do
  echo "[h3-stream] $m $(date '+%H:%M:%S')"
  python3 eval_local.py --model "$m" --part h3
done
echo "H3 STREAM DONE"
