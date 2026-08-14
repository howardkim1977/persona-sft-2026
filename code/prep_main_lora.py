# -*- coding: utf-8 -*-
"""
prep_main_lora.py — 본 실험 학습 데이터 변환 (사전등록 v0.2 준수)

조건별 유효 10,000쌍 → train 9,500 / valid 500 (시드 42 셔플, 조건 공통 규칙).
출력: lora_main/data_{C0,C1,C2}/{train,valid}.jsonl (mlx_lm 채팅 형식)
및 run별 설정 lora_main/config_{cond}_s{seed}.yaml.
"""

import json
import os
import random

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "lora_main")
SPLIT_SEED = 42
N_VALID = 500
ITERS = 4750          # 2에폭 x 9,500 / 배치 4
SEEDS = [42, 777]

rows = [json.loads(l) for l in open(os.path.join(BASE, "gen_main/results.jsonl"))]
os.makedirs(OUT, exist_ok=True)

for cond in ("C0", "C1", "C2"):
    valid = [r for r in rows if r["condition"] == cond and r["status"] == "valid"]
    # 결정적 정렬 후 셔플 (재현성: idx, replace_no 순)
    valid.sort(key=lambda r: (r["idx"], r["replace_no"]))
    assert len(valid) == 10000, f"{cond}: {len(valid)}"
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(valid)
    ddir = os.path.join(OUT, f"data_{cond}")
    os.makedirs(ddir, exist_ok=True)
    for name, part in (("valid", valid[:N_VALID]), ("train", valid[N_VALID:])):
        with open(os.path.join(ddir, f"{name}.jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": r["pair"]["instruction"]},
                    {"role": "assistant", "content": r["pair"]["response"]},
                ]}, ensure_ascii=False) + "\n")
    print(cond, "train", len(valid) - N_VALID, "valid", N_VALID)

# run별 mlx_lm 설정 (rank 16, alpha 32 -> scale = alpha/rank = 2.0)
for cond in ("C0", "C1", "C2"):
    for seed in SEEDS:
        cfg = f"""# 사전등록 v0.2 5절 고정 설정 (OSF fc8mn)
model: "mlx-community/EXAONE-3.5-2.4B-Instruct-bf16"
train: true
data: "{OUT}/data_{cond}"
seed: {seed}
num_layers: 16
batch_size: 4
iters: {ITERS}
learning_rate: 1.0e-4
max_seq_length: 2048
steps_per_report: 250
steps_per_eval: 250
save_every: 250
adapter_path: "{OUT}/adapters_{cond}_s{seed}"
fine_tune_type: lora
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 2.0    # alpha 32 / rank 16
"""
        with open(os.path.join(OUT, f"config_{cond}_s{seed}.yaml"), "w") as f:
            f.write(cfg)
print("설정 파일 6개 생성 완료")
