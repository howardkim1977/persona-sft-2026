# -*- coding: utf-8 -*-
"""
eval_local.py — 7개 평가 대상(원 모델 + 어댑터 6)의 로컬 응답 생성

파트:
  h1: 중립 프롬프트 500 → 개방형 응답 (temperature 0.7, max 512) [사전등록 6.1]
  h2: 하위집단 질의 400 → 응답 (temperature 0.0, max 512; 결정적 채점 대상)
  h3: KoBEST+KMMLU 6,561 MCQ → 알파벳 답 (temperature 0.0, max 8),
      선행 하네스의 t0 템플릿·parse_answer 이식 [사전등록 6.3]

실행: python3 eval_local.py --model base --part h1
      python3 eval_local.py --model C2_s42 --part h3
증분 저장 + 재개: eval_out/{part}_{model}.jsonl
"""

import argparse
import json
import os
import re
import time

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "mlx-community/EXAONE-3.5-2.4B-Instruct-bf16"
ADAPTER_DIR = os.path.join(BASE, "lora_main")
EVAL_DIR = os.path.join(BASE, "eval_sets")
OUT_DIR = os.path.join(BASE, "eval_out")

MODEL_KEYS = ["base", "C0_s42", "C0_s777", "C1_s42", "C1_s777",
              "C2_s42", "C2_s777"]

# --- ai-comparison run_eval.py 이식 (MCQ 형식·파싱 동일성 유지) ---
LETTERS = "ABCDE"
T0 = "다음 객관식 문제를 읽고, 정답 선택지의 알파벳 하나만 출력하시오.\n\n{body}\n\n정답:"
ANS_TAIL_RE = re.compile(r"정답[은는이:\s]*\**[\s(]*([A-E])(?![A-Za-z])")
ANS_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")
CIRCLED = {c: LETTERS[i] for i, c in enumerate("①②③④⑤")}


def build_body(item):
    if item["choices"] is None:
        return item["question"]
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n\n{opts}"


def parse_answer(content):
    content = (content or "").strip()
    tail = ANS_TAIL_RE.findall(content)
    if tail:
        return tail[-1]
    letters = ANS_RE.findall(content)
    if letters:
        return letters[-1]
    circled = [CIRCLED[c] for c in content if c in CIRCLED]
    return circled[-1] if circled else None


def best_adapter(cond_seed: str) -> str:
    """val loss 최저 체크포인트 선택 (사전등록 5절). 로그에서 val 손실 파싱."""
    log = os.path.join(ADAPTER_DIR, f"log_{cond_seed[:2]}_s{cond_seed.split('_s')[1]}.txt")
    adir = os.path.join(ADAPTER_DIR, f"adapters_{cond_seed[:2]}_s{cond_seed.split('_s')[1]}")
    vals = []
    for line in open(log):
        m = re.search(r"Iter (\d+): Val loss ([\d.]+)", line)
        if m:
            vals.append((float(m.group(2)), int(m.group(1))))
    if not vals:
        raise RuntimeError(f"val loss 기록 없음: {log}")
    best_iter = min(vals)[1]
    ckpt = os.path.join(adir, f"{best_iter:07d}_adapters.safetensors")
    if not os.path.exists(ckpt):
        # iter 1(학습 전) 최저인 비정상 경우 등 — 최종본 폴백, 기록
        print(f"경고: {ckpt} 없음, 최종 어댑터 사용")
        return adir
    # mlx_lm은 adapter_path 디렉토리의 adapters.safetensors를 로드하므로
    # 최적 체크포인트를 best/ 하위에 복사해 둔다.
    bdir = os.path.join(adir, "best")
    os.makedirs(bdir, exist_ok=True)
    import shutil
    shutil.copy(ckpt, os.path.join(bdir, "adapters.safetensors"))
    cfg_src = os.path.join(adir, "adapter_config.json")
    if os.path.exists(cfg_src):
        shutil.copy(cfg_src, os.path.join(bdir, "adapter_config.json"))
    print(f"{cond_seed}: best iter {best_iter} (val {min(vals)[0]:.3f})")
    return bdir


def load_items(part):
    if part == "h1":
        return [json.loads(l) for l in open(os.path.join(EVAL_DIR, "h1_prompts.jsonl"))]
    if part == "h2":
        return [json.loads(l) for l in open(os.path.join(EVAL_DIR, "h2_items.jsonl"))]
    return [json.loads(l) for l in open(os.path.join(EVAL_DIR, "h3_items.jsonl"))]


def item_key(part, i, item):
    if part == "h3":
        return item["uid"]
    return f"{part}:{i}"


def user_text(part, item):
    if part == "h1":
        return item["prompt"]
    if part == "h2":
        return item["instruction"]
    return T0.format(body=build_body(item))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_KEYS)
    ap.add_argument("--part", required=True, choices=["h1", "h2", "h3"])
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    os.makedirs(OUT_DIR, exist_ok=True)
    adapter = None
    if args.model != "base":
        adapter = best_adapter(args.model)
    model, tokenizer = load(MODEL_PATH, adapter_path=adapter,
                            tokenizer_config={"trust_remote_code": True})

    items = load_items(args.part)
    out_path = os.path.join(OUT_DIR, f"{args.part}_{args.model}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            done.add(json.loads(l)["key"])

    temp = 0.7 if args.part == "h1" else 0.0
    max_tok = 8 if args.part == "h3" else 512
    sampler = make_sampler(temp=temp)

    t0 = time.time()
    n_done = 0
    with open(out_path, "a") as fout:
        for i, item in enumerate(items):
            key = item_key(args.part, i, item)
            if key in done:
                continue
            msgs = [{"role": "user", "content": user_text(args.part, item)}]
            prompt = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True)
            text = generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tok, sampler=sampler)
            rec = {"key": key, "model": args.model, "response": text}
            if args.part == "h3":
                rec["pred"] = parse_answer(text)
                rec["gold"] = item["gold"]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                print(f"  {n_done} ({el:.0f}s, {n_done/el*60:.0f}/min)",
                      flush=True)
    print(f"완료: {out_path} (+{n_done})")


if __name__ == "__main__":
    main()
