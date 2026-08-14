# -*- coding: utf-8 -*-
"""
tasks.py
시드 과제 스키마와 조건별(C0/C1/C2) 생성 프롬프트 빌더.

설계 원칙 (DESIGN.md):
  - 과제 유형 목록은 세 조건이 완전히 공유하며, 페르소나 블록만 교체된다.
  - 응답은 어시스턴트 중립 문체(존댓말)로 고정한다. 페르소나는 지시(사용자
    요청)의 내용·관점·상황을 결정하는 요소이지, 응답의 화자가 아니다.
  - 과제 유형 8종은 파일럿에서 검증 후 사전등록 시 고정한다.
"""

import json
import random
import re
from typing import Dict, Optional

# --- 시드 과제 유형 (초안 v0.1, 사전등록 시 고정) ---
TASK_TYPES = {
    "정보질의": "일상 생활이나 관심 분야에 대해 사실적 정보를 묻는 질문",
    "조언요청": "본인의 상황을 설명하고 실질적 조언을 구하는 요청",
    "설명요청": "개념이나 절차를 자신의 눈높이에 맞게 설명해 달라는 요청",
    "일상대화": "가벼운 대화를 시작하거나 이어가는 발화",
    "글쓰기": "특정 목적의 글(문자, 편지, 안내문, 후기 등) 작성 요청",
    "계획수립": "일정, 여행, 행사, 목표 달성 등의 계획을 세워 달라는 요청",
    "추천요청": "선택지를 비교하거나 상황에 맞는 것을 추천해 달라는 요청",
    "문제해결": "생활 속 구체적 문제 상황의 해결 방법을 찾는 요청",
}

# 출력 형식 (세 조건 공통)
OUTPUT_SPEC = (
    '반드시 아래 JSON 형식으로만 출력하십시오. 다른 텍스트를 덧붙이지 마십시오.\n'
    '{"instruction": "<사용자의 요청 발화(한국어)>", '
    '"response": "<요청에 대한 충실한 모범 응답(한국어, 정중한 존댓말)>"}'
)

# --- C1 임의 페르소나: 인구통계 필드 균등 무작위 조합 ---
# 값 목록은 Nemotron 관측치와 동일한 어휘를 쓰되, 조합을 균등 추첨한다
# (접지 조건과의 차이는 '결합 분포의 현실성'뿐이도록 설계).
C1_FIELDS = {
    "sex": ["남자", "여자"],
    "age": list(range(19, 90)),
    "province": ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                 "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"],
    "occupation": ["사무직 회사원", "자영업자", "농업 종사자", "전문직", "판매 종사자",
                   "생산직 근로자", "주부", "학생", "무직", "서비스업 종사자",
                   "공무원", "프리랜서"],
    "education_level": ["초등학교", "중학교", "고등학교", "2~3년제 전문대학",
                        "4년제 대학교", "대학원(석사)", "대학원(박사)"],
}


def make_random_persona(rng: random.Random) -> Dict:
    """C1 임의 페르소나 1개 생성 (선행 silicon sampling 관행의 재현)."""
    return {k: rng.choice(v) for k, v in C1_FIELDS.items()}


def random_persona_block(p: Dict) -> str:
    return (f"[기본 정보] 성별: {p['sex']} / 나이: {p['age']}세 / "
            f"거주지: {p['province']} / 직업: {p['occupation']} / "
            f"최종학력: {p['education_level']}")


def build_generation_prompt(condition: str, task_type: str,
                            persona_block: Optional[str] = None) -> str:
    """조건별 지시-응답 쌍 생성 프롬프트.
    condition: "C0"(무페르소나) | "C1"(임의) | "C2"(접지)
    C1/C2는 persona_block 필수. 과제 지시문은 세 조건에서 동일 문장을 쓴다.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"미정의 과제 유형: {task_type}")
    task_desc = TASK_TYPES[task_type]

    if condition == "C0":
        who = "한국어 사용자 한 명"
        persona_part = ""
    elif condition in ("C1", "C2"):
        if not persona_block:
            raise ValueError(f"{condition}에는 persona_block이 필요합니다")
        who = "아래 인물"
        persona_part = f"\n다음은 가상의 인물 정보입니다.\n{persona_block}\n"
    else:
        raise ValueError(f"미정의 조건: {condition}")

    return (
        f"당신은 AI 어시스턴트의 학습 데이터를 만드는 데이터 생성기입니다.{persona_part}\n"
        f"{who}이 AI 어시스턴트에게 실제로 할 법한 '{task_type}' 유형의 요청을 1개 만드십시오.\n"
        f"과제 유형 설명: {task_desc}.\n"
        f"요청은 그 사람의 삶에서 자연스럽게 나올 구체적 상황을 담아야 하며, "
        f"말투와 어휘도 그 사람다워야 합니다. "
        f"이어서 그 요청에 대한 충실하고 도움이 되는 모범 응답을 작성하십시오.\n\n"
        f"{OUTPUT_SPEC}"
    )


def parse_pair(text: str) -> Optional[Dict]:
    """생성 출력에서 {"instruction","response"} JSON을 추출. 실패 시 None."""
    if not text:
        return None
    text = text.strip()
    # 코드펜스 제거
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        # strict=False: 문자열 내 비이스케이프 제어문자(실제 줄바꿈 등) 허용
        obj = json.loads(text[start:end + 1], strict=False)
    except json.JSONDecodeError:
        obj = _regex_fallback(text[start:end + 1])
        if obj is None:
            return None
    if not isinstance(obj, dict):
        return None
    if not obj.get("instruction") or not obj.get("response"):
        return None
    return {"instruction": str(obj["instruction"]), "response": str(obj["response"])}


def _regex_fallback(text: str) -> Optional[Dict]:
    """비이스케이프 내부 따옴표로 json.loads가 실패한 경우의 폴백 추출.
    키 구조("instruction" → "response" → 끝)에 기대어 값 구간을 잘라낸다."""
    m = re.search(
        r'"instruction"\s*:\s*"(.*?)"\s*,\s*"response"\s*:\s*"(.*)"\s*\}\s*$',
        text, re.DOTALL)
    if not m:
        return None
    unesc = lambda s: s.replace('\\n', '\n').replace('\\"', '"')
    return {"instruction": unesc(m.group(1)), "response": unesc(m.group(2))}
