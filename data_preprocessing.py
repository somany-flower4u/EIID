import numpy as np
import pandas as pd

#-----------------------------------------------------
# 1. 학교 위생상태 데이터 전처리
#-----------------------------------------------------
def Campus_Hygiene_Data_Processing(df):

    #-----------------------------------------------------
    # 1. 컬럼 리네임
    #-----------------------------------------------------
    rename_map = {
        # 기본정보
        '시도교육청': '시도교육청',
        '교육지원청': '교육지원청',
        '지역': '지역',
        '정보공시 학교코드': '학교코드',
        '학교명': '학교명',
        '학교급코드': '학교급',
        '설립구분': '설립구분',
        '제외여부': '제외여부',
        '상하반기구분': '상하반기',

        '자연환기 적합여부': '자연환기_적합',

        '적합여부': 'PM10_적합',
        '최종결과치': 'PM10_결과치',
        '적합여부.1': 'PM25_적합',
        '최종결과치.1': 'PM25_결과치',
        '적합여부.2': 'CO2_적합',
        '최종결과치.2': 'CO2_결과치',
        '적합여부.3': 'HCHO_적합',
        '최종결과치.3': 'HCHO_결과치',

        '정수기 적합여부': '정수기_적합',
        '지하수 적합여부': '지하수_적합',

        '적합여부.4': '총부유세균_적합',
        '최종결과치.4': '총부유세균_결과치',

        '적합여부.5': '낙하세균_적합',
        '1차 결과치': '낙하세균_1차', '2차 결과치': '낙하세균_2차', '3차 결과치': '낙하세균_3차',
        '적합여부.6': 'CO_적합',
        '1차 결과치.1': 'CO_1차', '2차 결과치.1': 'CO_2차', '3차 결과치.1': 'CO_3차',
        '적합여부.7': 'NO2_적합',
        '1차 결과치.2': 'NO2_1차', '2차 결과치.2': 'NO2_2차', '3차 결과치.2': 'NO2_3차',
        '적합여부.8': '라돈_적합',
        '1차 결과치.3': '라돈_1차', '2차 결과치.3': '라돈_2차', '3차 결과치.3': '라돈_3차',
        '적합여부.9': 'TVOC_적합',
        '1차 결과치.4': 'TVOC_1차', '2차 결과치.4': 'TVOC_2차', '3차 결과치.4': 'TVOC_3차',
        '적합여부.10': '벤젠_적합',
        '1차 결과치.5': '벤젠_1차', '2차 결과치.5': '벤젠_2차', '3차 결과치.5': '벤젠_3차',
        '적합여부.11': '톨루엔_적합',
        '1차 결과치.6': '톨루엔_1차', '2차 결과치.6': '톨루엔_2차', '3차 결과치.6': '톨루엔_3차',
        '적합여부.12': '에틸벤젠_적합',
        '1차 결과치.7': '에틸벤젠_1차', '2차 결과치.7': '에틸벤젠_2차', '3차 결과치.7': '에틸벤젠_3차',
        '적합여부.13': '자일렌_적합',
        '1차 결과치.8': '자일렌_1차', '2차 결과치.8': '자일렌_2차', '3차 결과치.8': '자일렌_3차',
        '적합여부.14': '스티렌_적합',
        '1차 결과치.9': '스티렌_1차', '2차 결과치.9': '스티렌_2차', '3차 결과치.9': '스티렌_3차',
        '적합여부.15': '석면_적합',
        '1차 결과치.10': '석면_1차', '2차 결과치.10': '석면_2차', '3차 결과치.10': '석면_3차',
        '적합여부.16': '오존_적합',
        '1차 결과치.11': '오존_1차', '2차 결과치.11': '오존_2차', '3차 결과치.11': '오존_3차',
        '적합여부.17': '진드기_적합',
        '1차 결과치.12': '진드기_1차', '2차 결과치.12': '진드기_2차', '3차 결과치.12': '진드기_3차',
    }

    df = df.rename(columns=rename_map)

    #-----------------------------------------------------
    # 2. 행 필터링 (제외 학교 & 결측 학교코드 제거)
    #-----------------------------------------------------
    if '제외여부' in df.columns:
        df = df[df['제외여부'] != 'Y'].copy()
        df = df.drop(columns=['제외여부', '제외사유'], errors='ignore')

    df = df.dropna(subset=['학교코드', '학교명'])

    #-----------------------------------------------------
    # 3. 타입 변환
    #-----------------------------------------------------
    # 결과치 → 숫자형
    result_cols = [c for c in df.columns if '결과치' in c or '1차' in c or '2차' in c or '3차' in c]
    for col in result_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 적합여부 → 이진(0/1)
    fit_cols = [c for c in df.columns if '적합' in c]
    for col in fit_cols:
        df[col] = df[col].map({'적합': 1, '부적합': 0, 'Y': 1, 'N': 0})

    #-----------------------------------------------------
    # 4. 파생 컬럼 생성 (다회차 측정 항목)
    #-----------------------------------------------------
    multi_items = [
        '낙하세균', 'CO', 'NO2', '라돈', 'TVOC', '벤젠',
        '톨루엔', '에틸벤젠', '자일렌', '스티렌', '석면', '오존', '진드기'
    ]

    new_cols = {}
    for item in multi_items:
        cols = [f'{item}_{i}차' for i in [1, 2, 3] if f'{item}_{i}차' in df.columns]
        if cols:
            new_cols[f'{item}_평균'] = df[cols].mean(axis=1, skipna=True)
            new_cols[f'{item}_최대'] = df[cols].max(axis=1, skipna=True)
            new_cols[f'{item}_측정횟수'] = df[cols].notna().sum(axis=1)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    #-----------------------------------------------------
    # 5. 핵심 집계 파생 변수
    #-----------------------------------------------------
    fit_cols = [c for c in df.columns if '적합' in c]

    df['검사_항목수'] = df[fit_cols].notna().sum(axis=1)
    df['부적합_항목수'] = df[fit_cols].apply(lambda x: (x == 0).sum(), axis=1)
    df['부적합_비율'] = df['부적합_항목수'] / df['검사_항목수'].replace(0, np.nan)
    df['미검사_항목수'] = df[fit_cols].isna().sum(axis=1)
    df['미검사_비율'] = df['미검사_항목수'] / len(fit_cols)

    #-----------------------------------------------------
    # 6. 주요 공기질 NaN → 지역+학교급 중앙값 대체
    #-----------------------------------------------------
    key_result_cols = [c for c in ['PM10_결과치', 'PM25_결과치', 'CO2_결과치',
                                    'HCHO_결과치', '총부유세균_결과치'] if c in df.columns]

    for col in key_result_cols:
        df[f'{col}_미검사'] = df[col].isna().astype(int)
        df[col] = df.groupby(['시도교육청', '학교급'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        # 그래도 남은 NaN (해당 그룹 전체가 NaN인 경우) → 전체 중앙값
        df[col] = df[col].fillna(df[col].median())

    return df

#-----------------------------------------------------
# 2. 방과후학교 운영현황 데이터 전처리
#-----------------------------------------------------
def Afterschool_Program_Data_Processing(df):
    """방과후학교 프로그램 데이터 전처리"""

    #-----------------------------------------------------
    # 1. 컬럼 리네임
    #-----------------------------------------------------
    rename_map = {
        # 기본정보
        '시도교육청': '시도교육청',
        '교육지원청': '교육지원청',
        '지역': '지역',
        '정보공시 학교코드': '학교코드',
        '학교명': '학교명',
        '학교급코드': '학교급',
        '설립구분': '설립구분',
        '제외여부': '제외여부',
        '제외사유': '제외사유',

        # 프로그램 현황
        '교과프로그램수': '교과_프로그램수',
        '교과수강학생수': '교과_수강학생수',
        '특기적성프로그램수': '특기적성_프로그램수',
        '특기적성수강학생수': '특기적성_수강학생수',
        '프로그램수(계)': '총_프로그램수',
        '수강학생수(계)': '총_수강학생수',
        '방과후학교참여학생수': '방과후_참여학생수',

        # 초등돌봄교실 (고등학교 데이터에서는 전부 0이지만, 범용성을 위해 유지)
        '초등돌봄교실 오후돌봄운영교실수': '돌봄_오후_교실수',
        '초등돌봄교실 오후돌봄참여학생수': '돌봄_오후_학생수',
        '초등돌봄교실 저녁돌봄운영교실수': '돌봄_저녁_교실수',
        '초등돌봄교실 저녁돌봄참여학생수': '돌봄_저녁_학생수',
        '초등돌봄교실방과후학교 연계형돌봄운영교실수': '돌봄_연계형_교실수',
        '초등돌봄교실방과후학교 연계형돌봄참여학생수': '돌봄_연계형_학생수',
        '초등돌봄참여학생중 특수교육대상학생수': '돌봄_특수교육_학생수',

        # 특수학교
        '특수학교종일반 학급수': '특수종일반_학급수',
        '특수학교종일반 참여학생수': '특수종일반_학생수',

        # 재정
        '방과후학교 수익자부담금액': '방과후_수익자부담',
        '방과후학교 수익자부담외지원금액': '방과후_외부지원',
        '초등돌봄교실 수익자부담금액': '돌봄_수익자부담',
        '초등돌봄교실 수익자부담외지원금액': '돌봄_외부지원',
    }

    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    #-----------------------------------------------------
    # 2. 행 필터링 (제외 학교 & 결측 학교코드 제거)
    #-----------------------------------------------------
    if '제외여부' in df.columns:
        df = df[df['제외여부'] != 'Y'].copy()
        df = df.drop(columns=['제외여부', '제외사유'], errors='ignore')

    df = df.dropna(subset=['학교코드', '학교명'])

    #-----------------------------------------------------
    # 3. 전부 0인 불필요 컬럼 제거
    #-----------------------------------------------------
    # 고등학교 데이터에서 초등돌봄/특수종일반 컬럼이 전부 0인 경우 제거
    zero_candidates = [
        '돌봄_오후_교실수', '돌봄_오후_학생수',
        '돌봄_저녁_교실수', '돌봄_저녁_학생수',
        '돌봄_연계형_교실수', '돌봄_연계형_학생수',
        '돌봄_특수교육_학생수',
        '특수종일반_학급수', '특수종일반_학생수',
        '돌봄_수익자부담', '돌봄_외부지원',
    ]

    drop_cols = [c for c in zero_candidates if c in df.columns and (df[c] == 0).all()]
    df = df.drop(columns=drop_cols)

    #-----------------------------------------------------
    # 4. 타입 변환 (방어적 처리)
    #-----------------------------------------------------
    numeric_cols = [
        '교과_프로그램수', '교과_수강학생수',
        '특기적성_프로그램수', '특기적성_수강학생수',
        '총_프로그램수', '총_수강학생수',
        '방과후_참여학생수',
        '방과후_수익자부담', '방과후_외부지원',
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        #-----------------------------------------------------
    # 5. 파생 변수 생성
    #-----------------------------------------------------
    new_cols = {}

    # ── 기존 유지 ──
    total_prog = df['총_프로그램수'].replace(0, np.nan)
    new_cols['교과_프로그램_비율'] = df['교과_프로그램수'] / total_prog
    new_cols['특기적성_프로그램_비율'] = df['특기적성_프로그램수'] / total_prog

    new_cols['교과_프로그램당_학생수'] = (
        df['교과_수강학생수'] / df['교과_프로그램수'].replace(0, np.nan)
    )
    new_cols['특기적성_프로그램당_학생수'] = (
        df['특기적성_수강학생수'] / df['특기적성_프로그램수'].replace(0, np.nan)
    )
    new_cols['전체_프로그램당_학생수'] = df['총_수강학생수'] / total_prog

    participation = df['방과후_참여학생수'].replace(0, np.nan)
    new_cols['수강_참여_비율'] = df['총_수강학생수'] / participation

    # ── 재정 지표 (기존 유지) ──
    total_budget = df['방과후_수익자부담'] + df['방과후_외부지원']
    new_cols['방과후_총재정'] = total_budget
    new_cols['방과후_외부지원_비율'] = (
        df['방과후_외부지원'] / total_budget.replace(0, np.nan)
    )
    new_cols['학생당_방과후_재정'] = total_budget / participation
    new_cols['프로그램당_재정'] = total_budget / total_prog

    # ══════════════════════════════════════════════════════
    # ── 추가: 프로그램 개설 중심 지표 ──
    # ══════════════════════════════════════════════════════

    # (A) 프로그램 다양성 지수 (교과 vs 특기적성 균형도, 0~1, 1이면 완전 균형)
    p_edu = new_cols['교과_프로그램_비율']
    p_ext = new_cols['특기적성_프로그램_비율']
    new_cols['프로그램_다양성지수'] = 1 - (p_edu - p_ext).abs()

    # (B) 프로그램 운영 여부 플래그
    new_cols['방과후_미운영'] = (df['총_프로그램수'] == 0).astype(int)

    # (C) 프로그램 규모 구간 (소/중/대)
    new_cols['프로그램_규모구간'] = pd.cut(
        df['총_프로그램수'],
        bins=[-1, 0, 10, 30, np.inf],
        labels=['미운영', '소규모', '중규모', '대규모']
    )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    #-----------------------------------------------------
    # 6. 결측값 처리 (파생변수에서 발생한 NaN)
    #-----------------------------------------------------
    # 0 나누기로 발생한 inf 처리
    df = df.replace([np.inf, -np.inf], np.nan)

    # 비율/파생 컬럼: 시도교육청 기준 중앙값 대체
    derived_cols = [
        '교과_프로그램_비율', '특기적성_프로그램_비율',
        '교과_프로그램당_학생수', '특기적성_프로그램당_학생수', '전체_프로그램당_학생수',
        '수강_참여_비율',
        '방과후_외부지원_비율', '학생당_방과후_재정', '프로그램당_재정',
    ]

    for col in derived_cols:
        if col in df.columns:
            # 원본 결측 여부 플래그
            df[f'{col}_결측'] = df[col].isna().astype(int)
            # 시도교육청별 중앙값으로 대체
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # 그래도 남은 NaN (해당 그룹 전체가 NaN) -> 전체 중앙값
            df[col] = df[col].fillna(df[col].median())

    #-----------------------------------------------------
    # 7. 지역 파싱 (시/도, 시/군/구 분리)
    #-----------------------------------------------------
    if '지역' in df.columns:
        df['지역'] = df['지역'].fillna('')
        df['시도'] = df['지역'].str.split(' ').str[0]
        df['시군구'] = df['지역'].str.split(' ').str[1:].str.join(' ')

    print(f"[INFO] 방과후학교 처리 완료: {df.shape[0]}행 * {df.shape[1]}열")
    print(f"[INFO] 제거된 전부-0 컬럼({len(drop_cols)}개): {drop_cols}")
    print(f"[INFO] 최종 컬럼: {list(df.columns)}")
    return df

#-----------------------------------------------------
# 3. 학교 보건실 운영현황 데이터 전처리
#-----------------------------------------------------
def Campus_Health_Office_Data_Processing(df):
    """보건실 현황 데이터 전처리"""

    #-----------------------------------------------------
    # 1. 컬럼 리네임
    #-----------------------------------------------------
    rename_map = {
        # 기본정보
        '시도교육청': '시도교육청',
        '교육지원청': '교육지원청',
        '지역': '지역',
        '정보공시 학교코드': '학교코드',
        '학교명': '학교명',
        '학교급코드': '학교급',
        '설립구분': '설립구분',
        '제외여부': '제외여부',
        '제외사유': '제외사유',

        # 보건실 현황
        '연간보건실 이용건수': '보건실_연간이용건수',
        '전체학생수': '전체학생수',
        '연간1인당 보건실이용건수': '보건실_1인당이용건수',
    }

    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    #-----------------------------------------------------
    # 2. 행 필터링 (제외 학교 & 결측 학교코드 제거)
    #-----------------------------------------------------
    if '제외여부' in df.columns:
        df = df[df['제외여부'] != 'Y'].copy()
        df = df.drop(columns=['제외여부', '제외사유'], errors='ignore')

    df = df.dropna(subset=['학교코드', '학교명'])

    #-----------------------------------------------------
    # 3. 전부 0인 불필요 컬럼 제거
    #-----------------------------------------------------
    numeric_check_cols = ['보건실_연간이용건수', '전체학생수', '보건실_1인당이용건수']
    drop_cols = [c for c in numeric_check_cols if c in df.columns and (df[c] == 0).all()]
    df = df.drop(columns=drop_cols)

    #-----------------------------------------------------
    # 4. 타입 변환 (방어적 처리)
    #-----------------------------------------------------
    numeric_cols = ['보건실_연간이용건수', '전체학생수', '보건실_1인당이용건수']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #-----------------------------------------------------
    # 5. 파생 변수 생성
    #-----------------------------------------------------
    new_cols = {}

    # (1) 1인당 이용건수 재계산 (원본 검증 및 결측 보완용)
    student_count = df['전체학생수'].replace(0, np.nan)
    new_cols['보건실_1인당이용건수_계산'] = df['보건실_연간이용건수'] / student_count

    # (2) 이용 강도 등급 (1인당 이용건수 기준 분위수 기반)
    usage_per_student = df['보건실_1인당이용건수']
    new_cols['보건실_이용강도'] = pd.cut(
        usage_per_student,
        bins=[-np.inf,
              usage_per_student.quantile(0.25),
              usage_per_student.quantile(0.50),
              usage_per_student.quantile(0.75),
              np.inf],
        labels=['낮음', '보통', '높음', '매우높음']
    )

    # (3) 시도교육청 평균 대비 이용 비율
    sido_mean = df.groupby('시도교육청')['보건실_1인당이용건수'].transform('mean')
    new_cols['시도평균대비_이용비율'] = df['보건실_1인당이용건수'] / sido_mean.replace(0, np.nan)

    # (4) 학교 규모 구분 (전체학생수 기준)
    new_cols['학교규모'] = pd.cut(
        df['전체학생수'],
        bins=[-np.inf, 100, 300, 600, 1000, np.inf],
        labels=['극소규모', '소규모', '중규모', '대규모', '과대규모']
    )

    # (5) 연간 이용건수 로그 변환 (분석용, 왜도 완화)
    new_cols['보건실_이용건수_log'] = np.log1p(df['보건실_연간이용건수'])

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    #-----------------------------------------------------
    # 6. 결측값 처리
    #-----------------------------------------------------
    # inf 처리
    df = df.replace([np.inf, -np.inf], np.nan)

    # 수치형 파생 컬럼: 시도교육청 기준 중앙값 대체
    derived_num_cols = [
        '보건실_1인당이용건수_계산',
        '시도평균대비_이용비율',
        '보건실_이용건수_log',
    ]

    for col in derived_num_cols:
        if col in df.columns:
            # 결측 플래그 생성
            df[f'{col}_결측'] = df[col].isna().astype(int)
            # 시도교육청별 중앙값 대체
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # 그룹 전체가 NaN인 경우 전체 중앙값으로 대체
            df[col] = df[col].fillna(df[col].median())

    # 원본 수치 컬럼 결측 처리
    for col in ['보건실_연간이용건수', '전체학생수', '보건실_1인당이용건수']:
        if col in df.columns:
            df[f'{col}_결측'] = df[col].isna().astype(int)
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # 범주형 파생 컬럼 결측 처리
    for col in ['보건실_이용강도', '학교규모']:
        if col in df.columns:
            df[col] = df[col].cat.add_categories('미분류') if hasattr(df[col], 'cat') else df[col]
            df[col] = df[col].fillna('미분류')

    #-----------------------------------------------------
    # 7. 지역 파싱 (시/도, 시/군/구 분리)
    #-----------------------------------------------------
    if '지역' in df.columns:
        df['지역'] = df['지역'].fillna('')
        df['시도'] = df['지역'].str.split(' ').str[0]
        df['시군구'] = df['지역'].str.split(' ').str[1:].str.join(' ')

    print(f"[INFO] 보건실 현황 처리 완료: {df.shape[0]}행 * {df.shape[1]}열")
    print(f"[INFO] 제거된 전부-0 컬럼({len(drop_cols)}개): {drop_cols}")
    print(f"[INFO] 최종 컬럼: {list(df.columns)}")
    return df

#-----------------------------------------------------
# 4. 학교 세입 데이터 전처리
#-----------------------------------------------------
def Campus_Income_Data_Processing(df):
    """학교 수입(세입) 현황 데이터 전처리"""

    #-----------------------------------------------------
    # 1. 컬럼 리네임
    #-----------------------------------------------------
    rename_map = {
        # 기본정보
        '시도교육청': '시도교육청',
        '교육지원청': '교육지원청',
        '지역': '지역',
        '정보공시 학교코드': '학교코드',
        '학교명': '학교명',
        '학교급코드': '학교급',
        '설립구분': '설립구분',
        '제외여부': '제외여부',
        '제외사유': '제외사유',

        # 구분
        '예결산구분': '예결산구분',
        '세입세출구분': '세입세출구분',

        # 수입 항목
        '정부이전수입': '수입_정부이전',
        '기타이전수입': '수입_기타이전',
        '학부모부담수입': '수입_학부모부담',
        '지원금수입': '수입_지원금',
        '행정활동수입': '수입_행정활동',
        '기타': '수입_기타',
    }

    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    #-----------------------------------------------------
    # 2. 행 필터링
    #-----------------------------------------------------
    # 제외 학교 제거
    if '제외여부' in df.columns:
        df = df[df['제외여부'] != 'Y'].copy()
        df = df.drop(columns=['제외여부', '제외사유'], errors='ignore')

    # 학교코드/학교명 결측 제거
    df = df.dropna(subset=['학교코드', '학교명'])

    # 예결산구분, 세입세출구분 확인 로그
    if '예결산구분' in df.columns:
        print(f"[INFO] 예결산구분 고유값: {df['예결산구분'].unique()}")
    if '세입세출구분' in df.columns:
        print(f"[INFO] 세입세출구분 고유값: {df['세입세출구분'].unique()}")

    # 세입(수입) 데이터만 필터링 (세출 데이터가 혼재되어 있을 경우)
    if '세입세출구분' in df.columns:
        income_keywords = ['세입', '수입']
        mask = df['세입세출구분'].str.contains('|'.join(income_keywords), na=False)
        if mask.any():
            df = df[mask].copy()
            print(f"[INFO] 세입 필터링 적용: {mask.sum()}행 유지")

    #-----------------------------------------------------
    # 3. 타입 변환
    #-----------------------------------------------------
    income_cols = [c for c in df.columns if c.startswith('수입_')]

    for col in income_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #-----------------------------------------------------
    # 4. 파생 변수 생성
    #-----------------------------------------------------
    new_cols = {}

    # (1) 총 수입
    existing_income_cols = [c for c in income_cols if c in df.columns]
    new_cols['수입_총계'] = df[existing_income_cols].sum(axis=1, skipna=True)

    # (2) 수입 항목별 구성비 (%)
    total = new_cols['수입_총계'].replace(0, np.nan)
    for col in existing_income_cols:
        short_name = col.replace('수입_', '')
        new_cols[f'구성비_{short_name}'] = (df[col] / total * 100).round(2)

    # (3) 자립도 지표: 정부 의존 비율 vs 자체 수입 비율
    gov_cols = ['수입_정부이전', '수입_지원금']
    self_cols = ['수입_학부모부담', '수입_행정활동', '수입_기타이전', '수입_기타']

    gov_exist = [c for c in gov_cols if c in df.columns]
    self_exist = [c for c in self_cols if c in df.columns]

    if gov_exist:
        new_cols['수입_정부의존합계'] = df[gov_exist].sum(axis=1, skipna=True)
        new_cols['정부의존비율'] = (new_cols['수입_정부의존합계'] / total * 100).round(2)

    if self_exist:
        new_cols['수입_자체합계'] = df[self_exist].sum(axis=1, skipna=True)
        new_cols['자체수입비율'] = (new_cols['수입_자체합계'] / total * 100).round(2)

    # (4) 학부모 부담 비율 (교육 불평등 핵심 지표)
    if '수입_학부모부담' in df.columns:
        new_cols['학부모부담비율'] = (df['수입_학부모부담'] / total * 100).round(2)

    # (5) 수입 총계 로그 변환 (왜도 완화)
    new_cols['수입_총계_log'] = np.log1p(new_cols['수입_총계'])

    # (6) 수입 규모 등급 (사분위 기반)
    income_total = new_cols['수입_총계']
    new_cols['수입규모등급'] = pd.cut(
        income_total,
        bins=[-np.inf,
              income_total.quantile(0.25),
              income_total.quantile(0.50),
              income_total.quantile(0.75),
              np.inf],
        labels=['소규모', '중소규모', '중대규모', '대규모']
    )

    # (7) 수입 다양성 지수 (허핀달 지수 역수 개념)
    # 항목별 비율의 제곱합이 낮을수록 수입원이 다양함
    ratio_cols_temp = []
    for col in existing_income_cols:
        ratio = df[col] / new_cols['수입_총계'].replace(0, np.nan)
        ratio_cols_temp.append(ratio.fillna(0))

    if ratio_cols_temp:
        ratio_df = pd.concat(ratio_cols_temp, axis=1)
        hhi = (ratio_df ** 2).sum(axis=1)
        new_cols['수입_집중도_HHI'] = hhi.round(4)
        new_cols['수입_다양성지수'] = (1 / hhi.replace(0, np.nan)).round(4)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    #-----------------------------------------------------
    # 5. 결측값 처리
    #-----------------------------------------------------
    # inf 처리
    df = df.replace([np.inf, -np.inf], np.nan)

    # 원본 수입 컬럼: 결측 = 해당 항목 수입 없음 → 0으로 대체
    for col in existing_income_cols:
        df[f'{col}_결측'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    # 파생 비율/지수 컬럼: 시도교육청별 중앙값 대체
    derived_num_cols = [
        '수입_총계', '수입_총계_log',
        '정부의존비율', '자체수입비율', '학부모부담비율',
        '수입_집중도_HHI', '수입_다양성지수',
    ]

    for col in derived_num_cols:
        if col in df.columns:
            df[f'{col}_결측'] = df[col].isna().astype(int)
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # 구성비 컬럼 결측 처리 (수입 총계가 0인 경우 발생)
    ratio_cols = [c for c in df.columns if c.startswith('구성비_')]
    for col in ratio_cols:
        df[col] = df[col].fillna(0)

    # 범주형 결측 처리
    if '수입규모등급' in df.columns:
        if hasattr(df['수입규모등급'], 'cat'):
            df['수입규모등급'] = df['수입규모등급'].cat.add_categories('미분류')
        df['수입규모등급'] = df['수입규모등급'].fillna('미분류')

    #-----------------------------------------------------
    # 6. 지역 파싱
    #-----------------------------------------------------
    if '지역' in df.columns:
        df['지역'] = df['지역'].fillna('')
        df['시도'] = df['지역'].str.split(' ').str[0]
        df['시군구'] = df['지역'].str.split(' ').str[1:].str.join(' ')

    print(f"[INFO] 수입 현황 처리 완료: {df.shape[0]}행 * {df.shape[1]}열")
    print(f"[INFO] 최종 컬럼 목록: {list(df.columns)}")
    return df

#-----------------------------------------------------
# 5. 학교 세출 데이터 전처리
#-----------------------------------------------------
def Campus_Outcome_Data_Processing(df):
    """학교 세출(지출) 현황 데이터 전처리"""

    #-----------------------------------------------------
    # 1. 컬럼 리네임
    #-----------------------------------------------------
    rename_map = {
        # 기본정보
        '시도교육청': '시도교육청',
        '교육지원청': '교육지원청',
        '지역': '지역',
        '정보공시 학교코드': '학교코드',
        '학교명': '학교명',
        '학교급코드': '학교급',
        '설립구분': '설립구분',
        '제외여부': '제외여부',
        '제외사유': '제외사유',

        # 구분
        '예결산구분': '예결산구분',
        '세입세출구분': '세입세출구분',

        # 세출 항목
        '인적자원운용': '세출_인적자원운용',
        '학생복지/교육격차해소': '세출_학생복지_교육격차해소',
        '기본적교육활동': '세출_기본적교육활동',
        '선택적교육활동': '세출_선택적교육활동',
        '교육활동지원': '세출_교육활동지원',
        '학교일반운영': '세출_학교일반운영',
        '학교시설확충': '세출_학교시설확충',
        '학교재무활동': '세출_학교재무활동',
    }

    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    #-----------------------------------------------------
    # 2. 행 필터링
    #-----------------------------------------------------
    # 제외 학교 제거
    if '제외여부' in df.columns:
        df = df[df['제외여부'] != 'Y'].copy()
        df = df.drop(columns=['제외여부', '제외사유'], errors='ignore')

    # 학교코드/학교명 결측 제거
    df = df.dropna(subset=['학교코드', '학교명'])

    # 구분값 확인 로그
    if '예결산구분' in df.columns:
        print(f"[INFO] 예결산구분 고유값: {df['예결산구분'].unique()}")
    if '세입세출구분' in df.columns:
        print(f"[INFO] 세입세출구분 고유값: {df['세입세출구분'].unique()}")

    # 세출(지출) 데이터만 필터링
    if '세입세출구분' in df.columns:
        expenditure_keywords = ['세출', '지출']
        mask = df['세입세출구분'].str.contains('|'.join(expenditure_keywords), na=False)
        if mask.any():
            df = df[mask].copy()
            print(f"[INFO] 세출 필터링 적용: {mask.sum()}행 유지")

    #-----------------------------------------------------
    # 3. 타입 변환
    #-----------------------------------------------------
    expenditure_cols = [c for c in df.columns if c.startswith('세출_')]

    for col in expenditure_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #-----------------------------------------------------
    # 4. 파생 변수 생성
    #-----------------------------------------------------
    new_cols = {}
    existing_exp_cols = [c for c in expenditure_cols if c in df.columns]

    # (1) 총 세출
    new_cols['세출_총계'] = df[existing_exp_cols].sum(axis=1, skipna=True)
    total = new_cols['세출_총계'].replace(0, np.nan)

    # (2) 세출 항목별 구성비 (%)
    for col in existing_exp_cols:
        short_name = col.replace('세출_', '')
        new_cols[f'구성비_{short_name}'] = (df[col] / total * 100).round(2)

    # -----------------------------------------------
    # (3) 기능별 그룹 합산 및 비율
    # -----------------------------------------------

    # 교육활동 직접투자: 기본적 + 선택적 교육활동
    edu_direct = ['세출_기본적교육활동', '세출_선택적교육활동']
    edu_direct_exist = [c for c in edu_direct if c in df.columns]
    if edu_direct_exist:
        new_cols['세출_교육직접투자'] = df[edu_direct_exist].sum(axis=1, skipna=True)
        new_cols['교육직접투자비율'] = (new_cols['세출_교육직접투자'] / total * 100).round(2)

    # 교육지원 간접투자: 교육활동지원 + 학생복지/교육격차해소
    edu_support = ['세출_교육활동지원', '세출_학생복지_교육격차해소']
    edu_support_exist = [c for c in edu_support if c in df.columns]
    if edu_support_exist:
        new_cols['세출_교육간접투자'] = df[edu_support_exist].sum(axis=1, skipna=True)
        new_cols['교육간접투자비율'] = (new_cols['세출_교육간접투자'] / total * 100).round(2)

    # 운영/관리비: 인적자원운용 + 학교일반운영
    operation = ['세출_인적자원운용', '세출_학교일반운영']
    operation_exist = [c for c in operation if c in df.columns]
    if operation_exist:
        new_cols['세출_운영관리비'] = df[operation_exist].sum(axis=1, skipna=True)
        new_cols['운영관리비비율'] = (new_cols['세출_운영관리비'] / total * 100).round(2)

    # 시설/재무: 학교시설확충 + 학교재무활동
    infra = ['세출_학교시설확충', '세출_학교재무활동']
    infra_exist = [c for c in infra if c in df.columns]
    if infra_exist:
        new_cols['세출_시설재무'] = df[infra_exist].sum(axis=1, skipna=True)
        new_cols['시설재무비율'] = (new_cols['세출_시설재무'] / total * 100).round(2)

    # -----------------------------------------------
    # (4) 핵심 비율 지표
    # -----------------------------------------------

    # 학생 직접 투자율 (교육직접 + 학생복지) / 총계
    student_direct = ['세출_기본적교육활동', '세출_선택적교육활동', '세출_학생복지_교육격차해소']
    student_direct_exist = [c for c in student_direct if c in df.columns]
    if student_direct_exist:
        new_cols['세출_학생직접투자'] = df[student_direct_exist].sum(axis=1, skipna=True)
        new_cols['학생직접투자율'] = (new_cols['세출_학생직접투자'] / total * 100).round(2)

    # 인건비 비율 (재정 경직성 지표)
    if '세출_인적자원운용' in df.columns:
        new_cols['인건비비율'] = (df['세출_인적자원운용'] / total * 100).round(2)

    # 학생복지 비율 (복지 투자 수준)
    if '세출_학생복지_교육격차해소' in df.columns:
        new_cols['학생복지비율'] = (df['세출_학생복지_교육격차해소'] / total * 100).round(2)

    # -----------------------------------------------
    # (5) 통계 파생 변수
    # -----------------------------------------------

    # 세출 총계 로그 변환
    new_cols['세출_총계_log'] = np.log1p(new_cols['세출_총계'])

    # 세출 규모 등급
    exp_total = new_cols['세출_총계']
    new_cols['세출규모등급'] = pd.cut(
        exp_total,
        bins=[-np.inf,
              exp_total.quantile(0.25),
              exp_total.quantile(0.50),
              exp_total.quantile(0.75),
              np.inf],
        labels=['소규모', '중소규모', '중대규모', '대규모']
    )

    # 지출 집중도 (HHI) 및 다양성 지수
    ratio_cols_temp = []
    for col in existing_exp_cols:
        ratio = df[col] / new_cols['세출_총계'].replace(0, np.nan)
        ratio_cols_temp.append(ratio.fillna(0))

    if ratio_cols_temp:
        ratio_df = pd.concat(ratio_cols_temp, axis=1)
        hhi = (ratio_df ** 2).sum(axis=1)
        new_cols['세출_집중도_HHI'] = hhi.round(4)
        new_cols['세출_다양성지수'] = (1 / hhi.replace(0, np.nan)).round(4)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    #-----------------------------------------------------
    # 5. 결측값 처리
    #-----------------------------------------------------
    # inf 처리
    df = df.replace([np.inf, -np.inf], np.nan)

    # 원본 세출 컬럼: 결측 = 해당 항목 지출 없음 → 0 대체
    for col in existing_exp_cols:
        df[f'{col}_결측'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    # 파생 수치 컬럼: 시도교육청별 중앙값 대체
    derived_num_cols = [
        '세출_총계', '세출_총계_log',
        '세출_교육직접투자', '세출_교육간접투자',
        '세출_운영관리비', '세출_시설재무', '세출_학생직접투자',
        '교육직접투자비율', '교육간접투자비율',
        '운영관리비비율', '시설재무비율',
        '학생직접투자율', '인건비비율', '학생복지비율',
        '세출_집중도_HHI', '세출_다양성지수',
    ]

    for col in derived_num_cols:
        if col in df.columns:
            df[f'{col}_결측'] = df[col].isna().astype(int)
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # 구성비 컬럼 결측 처리
    ratio_cols = [c for c in df.columns if c.startswith('구성비_')]
    for col in ratio_cols:
        df[col] = df[col].fillna(0)

    # 범주형 결측 처리
    if '세출규모등급' in df.columns:
        if hasattr(df['세출규모등급'], 'cat'):
            df['세출규모등급'] = df['세출규모등급'].cat.add_categories('미분류')
        df['세출규모등급'] = df['세출규모등급'].fillna('미분류')

    #-----------------------------------------------------
    # 6. 지역 파싱
    #-----------------------------------------------------
    if '지역' in df.columns:
        df['지역'] = df['지역'].fillna('')
        df['시도'] = df['지역'].str.split(' ').str[0]
        df['시군구'] = df['지역'].str.split(' ').str[1:].str.join(' ')

    print(f"[INFO] 세출 현황 처리 완료: {df.shape[0]}행 * {df.shape[1]}열")
    print(f"[INFO] 최종 컬럼 목록: {list(df.columns)}")
    return df

#-----------------------------------------------------
# 6. 학교 급식 데이터 전처리
#-----------------------------------------------------
def Campus_Lunch_Data_Processing(df):
    """학교 급식 현황 데이터 전처리"""
    print(f"[DEBUG] 실제 컬럼 목록:\n{df.columns.tolist()}")
    print(f"[DEBUG] 컬럼 수: {len(df.columns)}")
    #-----------------------------------------------------
    # 1. 컬럼 리네임
    #-----------------------------------------------------
    rename_map = {
        # 기본정보
        '시도교육청': '시도교육청',
        '교육지원청': '교육지원청',
        '지역': '지역',
        '정보공시 학교코드': '학교코드',
        '학교명': '학교명',
        '학교급코드': '학교급',
        '설립구분': '설립구분',
        '제외여부': '제외여부',
        '제외사유': '제외사유',

        # 운영방식
        '운영방식 - 급식종류': '급식종류',
        '운영방식 - 직영급식': '직영급식',
        '운영방식 - 전부위탁(업체명)': '전부위탁_업체명',
        '운영방식 - 일부위탁(업체명)': '일부위탁_업체명',

        # 학생 현황
        '전체학생수': '전체학생수',
        '급식학생수': '급식학생수',
        '급식비율': '급식비율',

        # 급식 인력
        '급식담당인력(명) - 영양(교)사': '영양교사수',
        '급식담당인력(명) - 조리사': '조리사수',
        '급식담당인력(명) - 조리원': '조리원수',

        # 배식장소
        '배식장소 - 배식장소(식당)': '배식_식당',
        '배식장소 - 배식장소(교실)': '배식_교실',
        '배식장소 - 배식장소(식당+교실)': '배식_식당교실',
        '배식장소 - 배식장소(기타)': '배식_기타',
    }

    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    #-----------------------------------------------------
    # 2. 행 필터링
    #-----------------------------------------------------
    # 제외 학교 제거
    if '제외여부' in df.columns:
        before_cnt = len(df)
        df = df[df['제외여부'] != 'Y'].copy()
        print(f"[INFO] 제외 학교 제거: {before_cnt} → {len(df)}행")
        df = df.drop(columns=['제외여부', '제외사유'], errors='ignore')

    # 학교코드/학교명 결측 제거
    df = df.dropna(subset=['학교코드', '학교명'])

    #-----------------------------------------------------
    # 3. 타입 변환
    #-----------------------------------------------------
    # 수치형 변환
    numeric_cols = [
        '전체학생수', '급식학생수', '급식비율',
        '영양교사수', '조리사수', '조리원수',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 배식장소 컬럼: 다양한 형태(O/X, Y/N, 1/0, 문자열) 대응
    place_cols = ['배식_식당', '배식_교실', '배식_식당교실', '배식_기타']
    for col in place_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].map(
                lambda x: 1 if x in ['O', 'Y', '1', '1.0', 'TRUE', 'YES'] else (
                    0 if x in ['X', 'N', '0', '0.0', 'FALSE', 'NO', '', 'NAN', 'NONE'] else np.nan
                )
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #-----------------------------------------------------
    # 4. 운영방식 파생 변수
    #-----------------------------------------------------
    # (1) 운영형태 통합 분류: 직영 / 전부위탁 / 일부위탁 / 미분류
    def classify_operation(row):
        has_direct = pd.notna(row.get('직영급식')) and str(row.get('직영급식', '')).strip() not in ['', 'nan', 'None']
        has_full = pd.notna(row.get('전부위탁_업체명')) and str(row.get('전부위탁_업체명', '')).strip() not in ['', 'nan', 'None']
        has_partial = pd.notna(row.get('일부위탁_업체명')) and str(row.get('일부위탁_업체명', '')).strip() not in ['', 'nan', 'None']

        if has_direct and not has_full and not has_partial:
            return '직영'
        elif has_full and not has_direct:
            return '전부위탁'
        elif has_partial:
            return '일부위탁'
        elif has_direct:
            return '직영'
        else:
            return '미분류'

    df['운영형태'] = df.apply(classify_operation, axis=1)

    # (2) 위탁 여부 이진 변수
    df['위탁여부'] = df['운영형태'].apply(lambda x: 1 if x in ['전부위탁', '일부위탁'] else 0)

    # (3) 급식종류 파싱 (중식, 석식, 조식 등 복수 가능)
    if '급식종류' in df.columns:
        df['급식종류_원본'] = df['급식종류'].copy()
        lunch_types = ['조식', '중식', '석식', '간식']
        for lt in lunch_types:
            df[f'급식_{lt}'] = df['급식종류'].astype(str).str.contains(lt, na=False).astype(int)

        # 급식 종류 수
        df['급식종류수'] = df[[f'급식_{lt}' for lt in lunch_types]].sum(axis=1)

    #-----------------------------------------------------
    # 5. 급식 인력 파생 변수
    #-----------------------------------------------------
    staff_cols = ['영양교사수', '조리사수', '조리원수']
    existing_staff = [c for c in staff_cols if c in df.columns]

    new_cols = {}

    # (1) 총 급식인력
    if existing_staff:
        new_cols['총급식인력'] = df[existing_staff].sum(axis=1, skipna=True)

    # (2) 조리인력 (조리사 + 조리원)
    cook_cols = [c for c in ['조리사수', '조리원수'] if c in df.columns]
    if cook_cols:
        new_cols['조리인력'] = df[cook_cols].sum(axis=1, skipna=True)

    # (3) 급식학생 1인당 인력 관련 비율
    if '총급식인력' in new_cols and '급식학생수' in df.columns:
        safe_students = df['급식학생수'].replace(0, np.nan)
        safe_staff = pd.Series(new_cols['총급식인력']).replace(0, np.nan)

        # 급식인력 1인당 담당 학생 수 (높을수록 인력 부족)
        new_cols['인력당학생수'] = (safe_students / safe_staff).round(2)

        # 학생 100명당 급식인력 수 (높을수록 인력 충분)
        new_cols['학생100명당인력'] = (new_cols['총급식인력'] / safe_students * 100).round(2)

    # (4) 조리인력 대비 영양교사 비율
    if '영양교사수' in df.columns and '조리인력' in new_cols:
        safe_cook = pd.Series(new_cols['조리인력']).replace(0, np.nan)
        new_cols['영양교사_조리인력비'] = (df['영양교사수'] / safe_cook).round(4)

    # (5) 영양교사 배치 여부 (1인 이상 = 배치됨)
    if '영양교사수' in df.columns:
        new_cols['영양교사배치'] = (df['영양교사수'] >= 1).astype(int)

    # (6) 인력 구성비
    if '총급식인력' in new_cols:
        safe_total_staff = pd.Series(new_cols['총급식인력']).replace(0, np.nan)
        for col in existing_staff:
            short = col.replace('수', '')
            new_cols[f'{short}_구성비'] = (df[col] / safe_total_staff * 100).round(2)

    #-----------------------------------------------------
    # 6. 급식 참여 파생 변수
    #-----------------------------------------------------
    if '전체학생수' in df.columns and '급식학생수' in df.columns:
        safe_total = df['전체학생수'].replace(0, np.nan)

        # (1) 급식비율 재계산 (원본 검증용)
        new_cols['급식비율_계산'] = (df['급식학생수'] / safe_total * 100).round(2)

        # (2) 비급식 학생수
        new_cols['비급식학생수'] = (df['전체학생수'] - df['급식학생수']).clip(lower=0)

        # (3) 비급식 비율
        new_cols['비급식비율'] = (new_cols['비급식학생수'] / safe_total * 100).round(2)

    # (4) 급식비율 등급
    if '급식비율' in df.columns:
        new_cols['급식참여등급'] = pd.cut(
            df['급식비율'],
            bins=[-np.inf, 50, 70, 90, 100, np.inf],
            labels=['매우낮음', '낮음', '보통', '높음', '완전급식']
        )

    #-----------------------------------------------------
    # 7. 배식장소 파생 변수
    #-----------------------------------------------------
    existing_place = [c for c in place_cols if c in df.columns]

    if existing_place:
        # (1) 배식장소 유형 통합
        def classify_place(row):
            s = row.get('배식_식당', 0) == 1
            c = row.get('배식_교실', 0) == 1
            sc = row.get('배식_식당교실', 0) == 1
            etc = row.get('배식_기타', 0) == 1

            if sc or (s and c):
                return '식당+교실'
            elif s:
                return '식당'
            elif c:
                return '교실'
            elif etc:
                return '기타'
            else:
                return '미분류'

        new_cols['배식장소유형'] = df.apply(classify_place, axis=1)

        # (2) 식당 보유 여부 (식당 또는 식당+교실)
        new_cols['식당보유'] = df.apply(
            lambda row: 1 if row.get('배식_식당', 0) == 1 or row.get('배식_식당교실', 0) == 1 else 0,
            axis=1
        )

    #-----------------------------------------------------
    # 8. 학교 규모 파생 변수
    #-----------------------------------------------------
    if '전체학생수' in df.columns:
        new_cols['학교규모'] = pd.cut(
            df['전체학생수'],
            bins=[-np.inf, 60, 300, 600, 1000, np.inf],
            labels=['극소규모', '소규모', '중규모', '대규모', '과대규모']
        )

        new_cols['전체학생수_log'] = np.log1p(df['전체학생수'])

    if '급식학생수' in df.columns:
        new_cols['급식학생수_log'] = np.log1p(df['급식학생수'])

    #-----------------------------------------------------
    # 9. 인력 적정성 지표
    #-----------------------------------------------------
    # 학교급별 권장 기준 대비 인력 적정성 판단
    # (일반적으로 조리원 1인당 학생 50~100명 기준)
    if '인력당학생수' in new_cols:
        ratio = pd.Series(new_cols['인력당학생수'], index=df.index)
        new_cols['인력적정성'] = pd.cut(
            ratio,
            bins=[-np.inf, 30, 50, 80, 100, np.inf],
            labels=['여유', '적정', '보통', '부족', '심각부족']
        )

    # 파생 변수 DataFrame 병합
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    #-----------------------------------------------------
    # 10. 결측값 처리
    #-----------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- 수치형 원본 컬럼 ---
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_결측'] = df[col].isna().astype(int)

    # 인력 컬럼: 결측 → 0 (미배치로 간주)
    for col in existing_staff:
        df[col] = df[col].fillna(0)

    # 학생수/급식비율: 시도교육청별 중앙값 대체
    student_cols = ['전체학생수', '급식학생수', '급식비율']
    for col in student_cols:
        if col in df.columns:
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # --- 파생 수치 컬럼 ---
    derived_numeric = [
        '총급식인력', '조리인력', '인력당학생수', '학생100명당인력',
        '영양교사_조리인력비', '급식비율_계산', '비급식학생수', '비급식비율',
        '전체학생수_log', '급식학생수_log',
    ]
    for col in derived_numeric:
        if col in df.columns:
            df[col] = df.groupby('시도교육청')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # --- 파생 구성비 컬럼 ---
    ratio_cols = [c for c in df.columns if c.endswith('_구성비')]
    for col in ratio_cols:
        df[col] = df[col].fillna(0)

    # --- 범주형 컬럼 ---
    cat_fill_map = {
        '급식참여등급': '미분류',
        '학교규모': '미분류',
        '인력적정성': '미분류',
        '배식장소유형': '미분류',
        '운영형태': '미분류',
    }
    for col, fill_val in cat_fill_map.items():
        if col in df.columns:
            if hasattr(df[col], 'cat') and fill_val not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(fill_val)
            df[col] = df[col].fillna(fill_val)

    # 배식장소 원본 이진 컬럼: 결측 → 0
    for col in existing_place:
        df[col] = df[col].fillna(0).astype(int)

    #-----------------------------------------------------
    # 11. 지역 파싱
    #-----------------------------------------------------
    if '지역' in df.columns:
        df['지역'] = df['지역'].fillna('')
        df['시도'] = df['지역'].str.split(' ').str[0]
        df['시군구'] = df['지역'].str.split(' ').str[1:].str.join(' ')

    #-----------------------------------------------------
    # 12. 최종 정리
    #-----------------------------------------------------
    # 불필요 컬럼 정리 (원본 위탁 업체명은 분석에 직접 사용 어려움, 보존은 하되 플래그화)
    if '전부위탁_업체명' in df.columns:
        df['전부위탁_있음'] = df['전부위탁_업체명'].notna().astype(int)
    if '일부위탁_업체명' in df.columns:
        df['일부위탁_있음'] = df['일부위탁_업체명'].notna().astype(int)

    print(f"[INFO] 급식 현황 처리 완료: {df.shape[0]}행 * {df.shape[1]}열")
    print(f"[INFO] 운영형태 분포:\n{df['운영형태'].value_counts()}")
    print(f"[INFO] 배식장소유형 분포:\n{df['배식장소유형'].value_counts()}")
    print(f"[INFO] 급식참여등급 분포:\n{df['급식참여등급'].value_counts()}")
    print(f"[INFO] 인력적정성 분포:\n{df['인력적정성'].value_counts()}")
    print(f"[INFO] 최종 컬럼 목록: {list(df.columns)}")
    return df

#-----------------------------------------------------
# 7. 학교 공간 데이터 전처리
#-----------------------------------------------------
def Campus_Space_Data_Processing(df_Campus_Space):
    # ============================================================
    # 1. 컬럼명 정리 (중복 컬럼명 해소: '기타', '기타.1')
    # ============================================================
    df_Campus_Space = df_Campus_Space.rename(columns={
        '기타'   : '교실_기타',       # 교실 관련 기타
        '기타.1' : '특별실_기타',     # 특별실(시청각·컴퓨터 등) 관련 기타
        '기타공간': '기타공간'         # 명시적 확인
    })

    print("\n[컬럼명 정리 완료]")
    print(df_Campus_Space.columns.tolist())

    # ============================================================
    # 2. 제외 대상 필터링
    # ============================================================
    before = len(df_Campus_Space)

    # '제외여부' 컬럼 값 분포 확인
    print(f"\n[제외여부 분포]\n{df_Campus_Space['제외여부'].value_counts(dropna=False)}")

    # 제외 대상 제거 (Y, 예, 1 등 다양한 형태 대응)
    exclude_values = ['Y', 'y', '예', '1', 1, True]
    df_Campus_Space = df_Campus_Space[
        ~df_Campus_Space['제외여부'].isin(exclude_values)
    ].copy()

    print(f"제외 처리: {before} → {len(df_Campus_Space)} ({before - len(df_Campus_Space)}건 제거)")

    # 제외 관련 컬럼 삭제
    df_Campus_Space.drop(columns=['제외여부', '제외사유'], inplace=True)

    # ============================================================
    # 3. 학교급코드 매핑 및 정리
    # ============================================================
    print(f"\n[학교급코드 원본 분포]\n{df_Campus_Space['학교급코드'].value_counts(dropna=False)}")

    school_level_map = {
        1: '초등학교', 2: '중학교', 3: '고등학교', 4: '특수학교',
        '1': '초등학교', '2': '중학교', '3': '고등학교', '4': '특수학교',
        '초': '초등학교', '중': '중학교', '고': '고등학교',
        '초등학교': '초등학교', '중학교': '중학교', '고등학교': '고등학교', '특수학교': '특수학교'
    }

    df_Campus_Space['학교급'] = df_Campus_Space['학교급코드'].map(school_level_map)

    # 매핑 안 된 값 확인
    unmapped = df_Campus_Space[df_Campus_Space['학교급'].isna()]['학교급코드'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ 매핑되지 않은 학교급코드: {unmapped}")
    else:
        print("✅ 학교급코드 매핑 완료")

    # ============================================================
    # 4. 공간 수치 컬럼 정의 및 숫자형 변환
    # ============================================================
    space_cols = [
        # --- 교실 영역 ---
        '일반교실', '교과교실', '특별교실', '수준별교실', '교실_기타',
        # --- 특별실 영역 ---
        '시청각실', '컴퓨터실', '멀티미디어실', '특별실_기타',
        # --- 교원·행정 영역 ---
        '교원지원공간', '관리행정공간',
        # --- 체육·보건 영역 ---
        '체육집회공간', '보건실', '학생탈의실', '학생샤워실',
        # --- 위생 영역 ---
        '남자화장실', '여자화장실', '공용화장실',
        # --- 급식·기숙·기타 ---
        '학생식당', '기숙사실수', '기타공간'
    ]

    for col in space_cols:
        if col in df_Campus_Space.columns:
            # 문자열 → 숫자 변환 (쉼표 제거, 빈문자열/'-' 처리)
            df_Campus_Space[col] = (
                df_Campus_Space[col]
                .astype(str)
                .str.strip()
                .str.replace(',', '', regex=False)
                .replace(['-', '', 'nan', 'None', '.'], np.nan)
            )
            df_Campus_Space[col] = pd.to_numeric(df_Campus_Space[col], errors='coerce')

    print(f"\n[수치 변환 후 dtypes]")
    print(df_Campus_Space[space_cols].dtypes)

    # ============================================================
    # 5. 결측치 처리 (공간 수 → 0 또는 NaN 유지 선택)
    # ============================================================
    print(f"\n[수치 컬럼 결측치 현황]")
    missing = df_Campus_Space[space_cols].isnull().sum()
    print(missing[missing > 0])

    # 공간 수는 '없음 = 0'으로 간주하여 0 대체
    df_Campus_Space[space_cols] = df_Campus_Space[space_cols].fillna(0).astype(int)

    print("✅ 결측치 → 0 대체 완료")

    # ============================================================
    # 6. 파생 변수 생성
    # ============================================================
    # 6-1. 교실 합계
    classroom_cols = ['일반교실', '교과교실', '특별교실', '수준별교실', '교실_기타']
    df_Campus_Space['교실_합계'] = df_Campus_Space[classroom_cols].sum(axis=1)

    # 6-2. 특별실 합계
    special_cols = ['시청각실', '컴퓨터실', '멀티미디어실', '특별실_기타']
    df_Campus_Space['특별실_합계'] = df_Campus_Space[special_cols].sum(axis=1)

    # 6-3. 위생시설 합계
    hygiene_cols = ['남자화장실', '여자화장실', '공용화장실']
    df_Campus_Space['화장실_합계'] = df_Campus_Space[hygiene_cols].sum(axis=1)

    # 6-4. 학생복지 공간 합계
    welfare_cols = ['보건실', '학생탈의실', '학생샤워실', '학생식당']
    df_Campus_Space['학생복지공간_합계'] = df_Campus_Space[welfare_cols].sum(axis=1)

    # 6-5. 전체 공간 합계
    df_Campus_Space['전체공간_합계'] = df_Campus_Space[space_cols].sum(axis=1)

    # 6-6. 기숙사 보유 여부
    df_Campus_Space['기숙사_보유'] = (df_Campus_Space['기숙사실수'] > 0).astype(int)

    print("\n[파생 변수 생성 완료]")
    print(df_Campus_Space[['교실_합계', '특별실_합계', '화장실_합계',
                            '학생복지공간_합계', '전체공간_합계', '기숙사_보유']].describe())

    # ============================================================
    # 7. 식별정보 처리 (대회 규정: 학교명·지역명 제출물 기재 금지)
    # ============================================================
    # 분석용에는 유지하되, 최종 제출 시 제거할 컬럼 목록 명시
    id_cols = ['학교명', '정보공시 학교코드', '교육지원청', '지역']
    print(f"\n⚠️ 최종 제출 시 제거 권장 식별 컬럼: {id_cols}")

    # ============================================================
    # 8. 설립구분 정리
    # ============================================================
    print(f"\n[설립구분 분포]\n{df_Campus_Space['설립구분'].value_counts(dropna=False)}")

    # ============================================================
    # 9. 이상치 탐지 (간단 점검)
    # ============================================================
    print("\n[이상치 점검 - 주요 컬럼 상위값]")
    for col in ['일반교실', '교실_합계', '전체공간_합계']:
        q1 = df_Campus_Space[col].quantile(0.25)
        q3 = df_Campus_Space[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1. * iqr
        outliers = df_Campus_Space[df_Campus_Space[col] > upper]
        print(f"  {col}: Q3={q3}, 상한={upper:.0f}, 이상치 {len(outliers)}건")

    return df_Campus_Space