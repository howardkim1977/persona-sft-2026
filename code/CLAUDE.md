# 접지 페르소나 합성 데이터 증강 연구 — 작업 지침

## 연구 맥락

통계 접지 페르소나(Nemotron-Personas-Korea) 조건화 합성 지시 데이터가 무페르소나·
임의 페르소나 대비 소형 한국어 모델(EXAONE-3.5-2.4B) 미세조정 결과를 개선하는지
검증하는 통제 실험이다. 합성 페르소나 연구 프로그램(persona-validation 등) 내
유일한 학습 연구다. 설계 상세는 [DESIGN.md](DESIGN.md). 목표: KCI 공학/정보통신
저널. 저자 김환·조근태.

## 설계 원칙 (프로그램 규약 계승)

- 처치 3조건(C0 무페르소나 / C1 임의 페르소나 / C2 접지 페르소나)에서 생성 데이터
  외 모든 것을 동일하게 유지한다. C1이 핵심 대조군(접지 여부 분리).
- 파일럿(조건당 200쌍) → OSF 사전등록 → 본 실험. 사전등록 전 본 실험 금지.
- 베이스라인 규율: 조정 전 원 모델을 상시 비교선에 포함한다.
- 재사용 코드: persona-validation의 map_persona.py(페르소나 매핑),
  ai-comparison의 평가 하네스(KoBEST/KMMLU)·심판 프로토콜.

## 절대 규칙

- 생성기는 K-EXAONE 2.0(Apache 2.0) 계열만 사용한다. **HCX-007 등 CLOVA Studio
  모델로 학습 데이터를 생성하지 않는다**(약관상 출력물의 언어모델 학습 사용 금지,
  2026-08-03 확인).
- Phase 0 관문 통과 전 본 생성 금지: (1) K-EXAONE 2.0 FriendliAI 서빙 확인
  (폴백: K-EXAONE 1.0), (2) 파일럿 비용 실측, (3) FriendliAI 크레딧 충전.
- API 키는 환경변수로만 주입. 코드·커밋 금지.
- 생성 파라미터(모델 버전·temperature·시드)는 사전등록 후 불변.
- 조건 간 품질 필터링·후처리는 반드시 동일 규칙으로 적용한다(교란 방지).

## 작성 규약

- 모든 설명·문서·주석은 격식체(평서체) 한국어. 객관적·간결.
- 가운데 점(U+00B7)은 목록 구분 외 본문 서술에 사용하지 않는다.
- 어휘: 노이즈·양상·기준 자료·베이스라인 등 선행 연구 규약 준수.

## 현재 단계

Phase 0 진행 중 (2026-08-03). 완료: 데이터 확보(persona-validation의
nemotron_personas_korea.csv 100만 행 재사용), map_persona.py 이식,
tasks.py(과제 8종+조건별 프롬프트+보강 파서), 스모크 테스트(smoke.py,
6페르소나 x 3과제 x 3모델, 결과 smoke/smoke_results.jsonl, 리뷰
smoke/smoke_review.md, 실비용 $0.34).

스모크 운영 교훈 (파일럿에 반영할 것):
- max_tokens 2000 이상 필수. V4-Flash는 reasoning 기본 on이라 명시적
  비활성화 필요(reasoning enabled false), gpt-oss는 effort low.
- Kimi는 JSON에 실제 줄바꿈/비이스케이프 따옴표 삽입 → strict=False +
  정규식 폴백 파서 필요. 파일럿에서 structured output(response_format) 검토.
- 결과 저장 시 raw_content 전문 저장(스모크는 2000자 절단해 재파싱 왜곡).
- 형식 성공(보정): V4-Flash 18/18, gpt-oss 18/18, Kimi 17/18(절단 1).
- 정성 관찰: 페르소나 말투 체화 Kimi > V4-Flash > gpt-oss(존댓말 평탄화).
  판정은 심판 채점으로 — 다음 작업.

C0/C1 스모크 + 심판 채점 + 파일럿 완료 (2026-08-04):
- 심판 채점(C2 54건 x sol+opus-5, judge_smoke.py): 품질 V4-Flash 1.24 ≈
  Kimi 1.25 > gpt-oss 0.89 (두 심판 순위 일치). fidelity는 전 모델 1.94~1.97
  천장 → 본 실험 채점은 척도 세분화(말투 사용역 차원 분리) 필요.
- **잠정 생성기 = DeepSeek-V4-Flash-0731** (Kimi와 품질 동률, 비용 1/50).
  사전등록 전 K-EXAONE 2.0 등재 시 동일 프로토콜 비교 후 최종 확정.
- 파일럿(pilot.py, 조건당 200쌍): 성공 600/600(재시도 38), 21분, **$0.085**.
  본 생성 3만 쌍 외삽 ≈ $4.3. distinct-2: C2 .890 > C1 .877 > C0 .800
  (H1 방향 신호, 단 C2-C1 격차 작음 — EMNLP 경고와 부합, H2 주가설 유지).
  C2 표본 실분포 확인: 60대+ 79/200, 비수도권 96/200 → H2 하위집단 분석
  검정력 확보 가능(본 생성 1만이면 고령 ~1,600).

다양성 지표 + LoRA 예행 완료 (2026-08-04):
- 다양성(metrics_diversity.py, pilot/diversity_report.json + diversity_notes.md):
  어휘 5지표(distinct-1/2/3, self-BLEU, 압축비) 전부 C2 > C1 > C0 일관.
  임베딩 분산(gemini-embedding-001, 768d)은 평탄~미세 역전 → 접지 효과는
  어휘·화용 층위. **사전등록에서 H1 = 어휘 다양성으로 조작적 정의**,
  임베딩 분산은 탐색적 지표로 병기.
- LoRA 예행(prep_lora_data.py + mlx_lm lora, adapters_pilot/): C2 180/20쌍,
  EXAONE-3.5-2.4B bf16, 200 iter 3.8분, 학습 파라미터 0.244%, peak 19.6GB.
  train loss 2.05→0.16, val 3.18→2.43(iter100)→3.10(iter200) = 소표본 과적합
  (본 실험은 에폭 조정으로 해소). 어댑터 생성 테스트 통과(노인 응대 문체 반영,
  단 영어 단어 누출 관찰 — 본 실험 품질 필터 고려).
- **Mac 학습 실측**: M5 Pro 48GB에서 0.88 it/s, 1,450 tok/s. 본 실험 외삽
  (1만 쌍 x 2에폭 x 6 runs) ≈ 9.5시간 → 로컬 하룻밤 가능, GPU 임대 $80~150
  절감 옵션. 스택 선택(MLX 로컬 vs 클라우드 HF+peft)은 사전등록 때 확정.
- mlx_lm generate는 EXAONE 로딩 시 trust_remote_code 프롬프트 발생 —
  echo "y" 파이프 또는 스크립트에서 trust_remote_code=True.

사전등록 v0.2 확정 (2026-08-04, 사용자 결정 반영):
docs/preregistration_draft.md. 확증 가설 H1(어휘 다양성 5지표, 4/5 기준),
H2(주가설, 하위집단 격차, 0-3 척도), H3(비열등 -2.0pp), 임베딩 분산 탐색적.
확정: 생성기 V4-Flash(2.0 등재돼도 본 실험 불변경), mlx_lm 0.31.3, 어댑터
공개(EXAONE 라이선스 3조건: EXAONE 접두 명칭/라이선스 사본/비상업 명시),
사람 채점 없음. 심판 4인 체제(주: gpt-5.6-sol + claude-opus-5 / 보조:
gpt-5.2 + HCX-007 — 한국계 관점 다양성, 평가 전용이라 CLOVA 약관의 학습·개발
금지와 구분, 채점은 CLOVA_STUDIO_API_KEY 경로 재사용). 본 실험 예산 ~$90-170.

OSF 등록 제출 완료 (2026-08-04): 프로젝트 osf.io/fc8mn, Open-Ended
Registration, 등록문 영어 전문, 즉시 공개 선택. 상태 Pending registration
approval — 사용자 이메일의 승인 링크 클릭 시 즉시 확정(미클릭 시 48시간 후
자동 승인). 승인 확정 전 본 생성 금지.

다음 작업: 등록 승인 확인 후 본 실험:
(1) 본 생성 3만 쌍(V4-Flash, ~$5, 수 시간), (2) LoRA 6 runs(로컬 ~9.5시간),
(3) 평가 4축(H1 프롬프트 500 + H2 400문항 x 7모델 x 3심판 + H3 벤치마크),
(4) 분석·집필. **사전등록 완료 전 본 생성 금지(절대 규칙).**
