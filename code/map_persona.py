# -*- coding: utf-8 -*-
"""
map_persona.py  (persona-validation에서 이식, 2026-08-03)
NVIDIA Nemotron-Personas-Korea 레코드를 본 연구 규약으로 매핑한다.

원본: ~/Research/persona-validation/map_persona.py (검증 완료된 26필드 스키마).
이식 변경점:
  - KISDI 정합용 매핑(교육수준 6구분, 시도 표준명)은 층화 보조 용도로 유지.
  - build_persona_prompt 기본 include를 범용 지시 생성용으로 조정
    (arts_persona 시나리오 특화 제거, professional_persona 추가).
"""

from typing import Dict, List, Optional

# 검증된 실제 컬럼명 (오타·누락 방지용 단일 출처)
PERSONA_NARRATIVE_COLS = [
    "professional_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "family_persona", "persona",
]
ATTRIBUTE_COLS = [
    "cultural_background", "skills_and_expertise", "skills_and_expertise_list",
    "hobbies_and_interests", "hobbies_and_interests_list", "career_goals_and_ambitions",
]
DEMOGRAPHIC_COLS = [
    "sex", "age", "marital_status", "military_status", "family_type",
    "housing_type", "education_level", "bachelors_field", "occupation",
    "district", "province", "country",
]
ALL_COLS = ["uuid"] + PERSONA_NARRATIVE_COLS + ATTRIBUTE_COLS + DEMOGRAPHIC_COLS  # 26

AGE = {1: "10세미만", 2: "10대", 3: "20대", 4: "30대",
       5: "40대", 6: "50대", 7: "60대", 8: "70대이상"}

SEX_MAP = {"남자": "남성", "여자": "여성"}

EDUCATION_MAP = {
    "초등학교": "초졸이하",
    "중학교": "중졸",
    "고등학교": "고졸",
    "2~3년제 전문대학": "전문대졸",
    "4년제 대학교": "대졸",
    "대학원(석사)": "대학원졸", "석사": "대학원졸",
    "대학원(박사)": "대학원졸", "박사": "대학원졸", "대학원": "대학원졸",
}

PROVINCE_TO_SIDO = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시",
    "경기": "경기도", "강원": "강원특별자치도",
    "충북": "충청북도", "충남": "충청남도",
    "충청북": "충청북도", "충청남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도",
    "전라북": "전북특별자치도", "전라남": "전라남도",
    "경상북": "경상북도", "경북": "경상북도",
    "경상남": "경상남도", "경남": "경상남도",
    "제주": "제주특별자치도",
}


def age_to_code(age: Optional[int]) -> Optional[int]:
    """Nemotron 정수 연령(19~99) → 연령대 코드."""
    if age is None:
        return None
    age = int(age)
    if age < 10:
        return 1
    if age < 20:
        return 2
    if age >= 70:
        return 8
    return (age // 10) + 1


def build_persona_prompt(record: Dict,
                         include: Optional[List[str]] = None) -> str:
    """지시 데이터 생성용 페르소나 프롬프트 블록 생성.
    범용 지시 생성이 목적이므로 기본 포함:
    persona(요약)+cultural_background+hobbies_and_interests+professional_persona.
    프롬프트에는 원자료의 자연어 값을 그대로 쓴다(남자/여자, 실연령 등).
    """
    if include is None:
        include = ["persona", "cultural_background",
                   "hobbies_and_interests", "professional_persona"]

    parts: List[str] = []
    parts.append(
        f"[기본 정보] 성별: {record.get('sex', '')} / 나이: {record.get('age', '')}세 / "
        f"거주지: {record.get('province', '')} {record.get('district', '')} / "
        f"직업: {record.get('occupation', '')} / "
        f"최종학력: {record.get('education_level', '')}"
    )
    labels = {
        "persona": "요약", "cultural_background": "성장·생활배경",
        "hobbies_and_interests": "취미·관심사", "arts_persona": "문화·미디어 성향",
        "professional_persona": "직업 성향", "career_goals_and_ambitions": "목표",
    }
    for col in include:
        val = record.get(col)
        if val:
            parts.append(f"[{labels.get(col, col)}] {val}")
    return "\n".join(parts)


def map_persona(record: Dict,
                prompt_include: Optional[List[str]] = None,
                strict: bool = False) -> Dict:
    """단일 페르소나 레코드 → 표준 매핑 결과.
    반환: uuid, segments(층화·하위집단 분석용), raw, persona_prompt.
    """
    age = record.get("age")
    sex_raw = record.get("sex")
    prov_raw = (record.get("province") or "").strip()
    edu_raw = (record.get("education_level") or "").strip()

    sido = PROVINCE_TO_SIDO.get(prov_raw)
    edu_std = EDUCATION_MAP.get(edu_raw)
    age_code = age_to_code(age)
    warnings = []
    if sido is None:
        warnings.append(f"province 미확인: '{prov_raw}'")
        sido = prov_raw
    if edu_std is None:
        warnings.append(f"education_level 미확인: '{edu_raw}'")
        edu_std = edu_raw

    # 주의: Nemotron CSV는 UTF-8 BOM을 포함하므로 첫 컬럼 키가 '﻿uuid'로
    # 읽힌다(2026-08-06 발견). 두 키를 모두 확인한다.
    uid = record.get("uuid") or record.get("﻿uuid")

    result = {
        "uuid": uid,
        "segments": {
            "성별": SEX_MAP.get(sex_raw, sex_raw),
            "연령대": age_code,
            "연령대_라벨": AGE.get(age_code),
            "시도": sido,
            "직업": (record.get("occupation") or "").strip(),
            "교육수준": edu_std,
        },
        "raw": {
            "sex": sex_raw, "age": age,
            "province": prov_raw, "district": record.get("district"),
            "education_level": edu_raw, "occupation": record.get("occupation"),
            "marital_status": record.get("marital_status"),
            "household": record.get("family_type"),
        },
        "persona_prompt": build_persona_prompt(record, include=prompt_include),
    }
    if warnings:
        result["warnings"] = warnings
        if strict:
            raise ValueError("; ".join(warnings))
    return result
