# 상용 API 약관 검증 기록 (Endnote 2 근거)

본 연구가 상용 폐쇄 API를 학습 데이터 생성기에서 배제한 근거다. 각 항목은
검증일 기준으로 확인한 원문 조항과 출처를 기록한다. 전문(全文) 재배포는
각 사업자의 저작권 정책을 고려해 생략하고, 판단의 근거가 된 조항과 접근
경로를 남긴다.

## 1. NAVER CLOVA Studio (검증 2026-08-03)

출처: CLOVA Studio 서비스 이용약관 (네이버클라우드 공개 PDF)
https://xv-ncloud.pstatic.net/images/provision/250106_민간_CLOVAStudio서비스이용약관_(CLEAN)_1735285423631.pdf

해당 조항(요지): 이용자는 회사가 "허용하는 범위 밖에서 언어모델의 학습
또는 개발하거나 제3자에게 본 서비스의 결과물을 제공하여 제3자가 언어모델의
학습 또는 개발할 수 있게 해서는 안 됩니다."

판단: HCX-007의 출력으로 타 언어모델(EXAONE)을 학습시키는 것은 금지 대상에
해당한다. 따라서 생성기에서 배제했다. 다만 평가 전용 채점(점수가 학습·모델
선택에 개입하지 않음)은 학습·개발과 구분되므로 보조심판으로만 사용했다.

## 2. Google Gemini API (검증 2026-08-03)

출처 1: Gemini API Additional Terms of Service
https://ai.google.dev/gemini-api/terms
해당 조항(요지): "You may not use the Services to develop models that
compete with the Services."

출처 2: Google Cloud Service Specific Terms (Vertex AI)
https://cloud.google.com/terms/service-terms
해당 조항(요지): "Customer will not ... use an AI/ML Service or Generated
Output to develop a similar or competing product or service."

판단: 출력물(Generated Output)을 명시적으로 지목하므로 생성기 사용 불가.

## 3. OpenAI / Anthropic 상용 API (검증 2026-08-03)

두 사업자의 이용약관도 경쟁 모델 개발을 위한 출력물 사용을 제한한다. 위
두 사례와 동일한 사유로 생성기 후보에서 배제했으며, 평가 심판 역할로만
사용했다(gpt-5.6-sol, gpt-5.2, claude-opus-5).

## 4. 채택한 오픈웨이트 모델의 라이선스 (licenses/ 참조)

- DeepSeek-V4-Flash-0731: MIT (생성기로 채택). 출력물 사용 제한 없음.
- EXAONE-3.5-2.4B-Instruct: EXAONE AI Model License 1.1-NC (미세조정 대상).
  2.1(a)(b)(c) 연구 목적 사용·공표·파생물 생성 허용, 2.1(d) 파생물 배포
  허용(라이선스 사본 동봉 조건), 명칭은 "EXAONE"으로 시작해야 함,
  3.1 상업적 이용 금지. 본 연구는 연구 목적이므로 요건을 충족한다.
- 참고: K-EXAONE 2.0(Apache 2.0)은 등록 시점에 공개 API가 없어 제외했다.
