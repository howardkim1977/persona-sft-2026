# -*- coding: utf-8 -*-
"""
build_package.py — 재현 패키지 조립 (원고의 공개 주장 이행)

포함: 코드 전량, 분석 스크립트·결과, 생성 데이터, 평가 세트·응답, 심판
원시 채점 로그, 논문·그림, 라이선스, 약관 검증 기록, 어댑터(EXAONE 명명
규칙 준수), MANIFEST(SHA-256).
제외: Nemotron 원본 CSV(4.1GB, NVIDIA가 공개 배포), API 키, 파일럿 원자료는
요약만.
"""

import hashlib
import json
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(BASE, "package")

COPY = {
    "code": ["map_persona.py", "tasks.py", "gen_main.py", "prep_eval.py",
             "eval_local.py", "judge_main.py", "prep_main_lora.py",
             "prep_lora_data.py", "metrics_diversity.py", "smoke.py",
             "smoke_c01.py", "judge_smoke.py", "pilot.py", "build_package.py",
             "eval_all.sh", "eval_h3_stream.sh", "CLAUDE.md", "DESIGN.md"],
    "analysis": ["analysis/h1_analysis.py", "analysis/h2_analysis.py",
                 "analysis/manip_check.py", "analysis/exploratory.py",
                 "analysis/repair_uuid.py", "analysis/verify_manuscript.py",
                 "analysis/h1_confirmatory.json",
                 "analysis/h2_confirmatory.json",
                 "analysis/h3_confirmatory.txt", "analysis/manip_check.json",
                 "analysis/exploratory.json",
                 "analysis/register_age_exploratory.txt"],
    "data": ["gen_main/results_uuid.jsonl", "gen_main/manifest.json",
             "gen_main/c2_train_personas.jsonl",
             "gen_main/c2_reserve_personas.jsonl",
             "gen_main/eval_personas.jsonl", "gen_main/c1_personas.jsonl"],
    "eval": ["eval_sets/h1_prompts.jsonl", "eval_sets/h2_items_uuid.jsonl",
             "eval_sets/h3_items.jsonl"],
    "judge": ["judge_out/scores.jsonl"],
    "paper": ["paper/draft_ksaforum_en.md", "paper/make_figure1.py",
              "paper/make_figure_design.py", "paper/build_docx.py",
              "paper/similarity_check.py",
              "paper/figures/figure_design.png",
              "paper/figures/figure1_register_gap.png",
              "paper/figures/figure_design.pdf",
              "paper/figures/figure1_register_gap.pdf"],
    "docs": ["docs/preregistration_draft.md"],
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1) 파일 복사
    for sub, files in COPY.items():
        d = os.path.join(PKG, sub)
        os.makedirs(d, exist_ok=True)
        for rel in files:
            src = os.path.join(BASE, rel)
            if not os.path.exists(src):
                print("  (누락)", rel)
                continue
            shutil.copy2(src, os.path.join(d, os.path.basename(rel)))

    # 2) 평가 응답 (7모델 x 3파트)
    d = os.path.join(PKG, "eval", "responses")
    os.makedirs(d, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(BASE, "eval_out"))):
        shutil.copy2(os.path.join(BASE, "eval_out", f), os.path.join(d, f))

    # 3) 어댑터 — EXAONE 라이선스 명명 규칙 준수
    ad = os.path.join(PKG, "adapters")
    os.makedirs(ad, exist_ok=True)
    for cond in ("C0", "C1", "C2"):
        for seed in ("42", "777"):
            src = os.path.join(BASE, f"lora_main/adapters_{cond}_s{seed}/best")
            if not os.path.exists(src):
                continue
            name = f"EXAONE-3.5-2.4B-personaSFT-{cond}-seed{seed}-adapter"
            dst = os.path.join(ad, name)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            shutil.copy2(
                os.path.join(PKG, "licenses",
                             "EXAONE_AI_Model_License_1.1_NC.txt"),
                os.path.join(dst, "LICENSE.txt"))
            open(os.path.join(dst, "NOTICE.txt"), "w").write(
                f"{name}\n\n"
                "This is a LoRA adapter derived from "
                "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct.\n"
                "Distributed under the EXAONE AI Model License Agreement "
                "1.1-NC (see LICENSE.txt).\n"
                "Research and non-commercial use only. Commercial use is "
                "prohibited.\n")

    # 4) MANIFEST
    lines = []
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in sorted(files):
            if f == "MANIFEST.sha256":
                continue
            p = os.path.join(root, f)
            lines.append(f"{sha256(p)}  {os.path.relpath(p, PKG)}")
    open(os.path.join(PKG, "MANIFEST.sha256"), "w").write("\n".join(lines))

    total = 0
    for r, ds, fs in os.walk(PKG):
        ds[:] = [d for d in ds if d != ".git"]
        total += sum(os.path.getsize(os.path.join(r, f)) for f in fs)
    print(f"패키지 파일 {len(lines)}개, {total/1e6:.0f}MB → {PKG}")


if __name__ == "__main__":
    main()
