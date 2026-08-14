# -*- coding: utf-8 -*-
"""
prep_lora_data.py — 파일럿 쌍을 mlx_lm LoRA 채팅 형식으로 변환 (예행용)

C2 200쌍 → train 180 / valid 20 (시드 고정 셔플).
출력: lora_data/train.jsonl, lora_data/valid.jsonl
형식: {"messages": [{"role":"user",...},{"role":"assistant",...}]}
"""

import json
import os
import random

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "lora_data")
SEED = 780

rows = [json.loads(l) for l in open(os.path.join(BASE, "pilot/pilot_pairs.jsonl"))]
c2 = [r for r in rows if r["condition"] == "C2" and r["ok"]]
rng = random.Random(SEED)
rng.shuffle(c2)

os.makedirs(OUT, exist_ok=True)
split = int(len(c2) * 0.9)
for name, part in (("train", c2[:split]), ("valid", c2[split:])):
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for r in part:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": r["pair"]["instruction"]},
                {"role": "assistant", "content": r["pair"]["response"]},
            ]}, ensure_ascii=False) + "\n")
    print(name, len(part))
