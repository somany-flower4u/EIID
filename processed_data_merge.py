import numpy as np
import pandas as pd

def merge_data(df_Campus_Hygiene,df_Afterschool_Program,df_Campus_HealthOffice,df_Campus_Income,df_Campus_Outcome,df_Campus_Lunch,df_Campus_Space):

    # ================================================================
    # PHASE 1: 병합 전 진단
    # ================================================================

    def diagnose_datasets(dataset_dict):
        """병합 전 각 데이터셋의 키 호환성 및 품질 진단"""

        print("=" * 10)
        print("PHASE 1: 데이터셋 간 병합 호환성 진단")
        print("=" * 10)

        # 1-1. 기본 현황
        print("\n[1-1] 데이터셋별 기본 현황")
        print(f"{'데이터셋':<20} {'행수':>8} {'열수':>8} {'키컬럼':>15} {'고유학교수':>12}")
        print("-" * 10)

        key_col_candidates = ['학교코드', '정보공시 학교코드']
        key_info = {}

        for name, df in dataset_dict.items():
            key_col = None
            for kc in key_col_candidates:
                if kc in df.columns:
                    key_col = kc
                    break

            n_unique = df[key_col].nunique() if key_col else 'N/A'
            print(f"{name:<20} {len(df):>8} {len(df.columns):>8} {str(key_col):>15} {str(n_unique):>12}")
            key_info[name] = {'key_col': key_col, 'n_unique': n_unique, 'n_rows': len(df)}

        # 1-2. 학교코드 교집합 분석
        print("\n[1-2] 학교코드 교집합 분석")
        code_sets = {}
        for name, df in dataset_dict.items():
            kc = key_info[name]['key_col']
            if kc:
                code_sets[name] = set(df[kc].dropna().unique())

        all_names = list(code_sets.keys())

        # ★ 안전 초기화: 데이터셋이 부족할 경우 대비
        common_all = set()

        if len(all_names) >= 2:
            common_all = set.intersection(*code_sets.values())
            print(f"  전체 교집합 (모든 데이터셋 공통): {len(common_all)}개 학교")

            # 쌍별 교집합
            print(f"\n  쌍별 교집합 행렬:")
            header = f"{'':>20}" + "".join(f"{n[:8]:>10}" for n in all_names)
            print(header)
            for n1 in all_names:
                row = f"{n1[:18]:>20}"
                for n2 in all_names:
                    overlap = len(code_sets[n1] & code_sets[n2])
                    row += f"{overlap:>10}"
                print(row)
        elif len(all_names) == 1:
            common_all = code_sets[all_names[0]]
            print(f"  ⚠️ 데이터셋이 1개뿐, 교집합 분석 불가 (학교 {len(common_all)}개)")
        else:
            print("  ⚠️ 유효한 키 컬럼을 가진 데이터셋이 없습니다.")

        # 1-3. 중복 행 진단 (같은 학교코드가 여러 행인 경우)
        print("\n[1-3] 학교코드 중복 행 진단 (1:N 관계 탐지)")
        for name, df in dataset_dict.items():
            kc = key_info[name]['key_col']
            if kc:
                n_unique_schools = df[kc].nunique()
                dup_ratio = len(df) / n_unique_schools if n_unique_schools > 0 else 0
                dup_schools = df[df.duplicated(subset=[kc], keep=False)][kc].nunique()
                print(f"  {name:<20}: 행/학교 비율 = {dup_ratio:.2f}, 중복 학교 {dup_schools}개")
            else:
                print(f"  {name:<20}: ⚠️ 키 컬럼 없음, 진단 불가")

        # 1-4. 공통 컬럼 충돌 진단 (병합 시 충돌 가능성)
        print("\n[1-4] 공통 컬럼 충돌 진단")
        base_cols = {'시도교육청', '교육지원청', '지역', '학교코드', '정보공시 학교코드',
                    '학교명', '학교급', '학교급코드', '설립구분', '시도', '시군구',
                    '예결산구분', '세입세출구분'}

        conflict_found = False
        for i, n1 in enumerate(all_names):
            for n2 in all_names[i+1:]:
                common_cols = set(dataset_dict[n1].columns) & set(dataset_dict[n2].columns)
                conflict_cols = common_cols - base_cols
                if conflict_cols:
                    print(f"  ⚠️ {n1} ↔ {n2}: 충돌 컬럼 {len(conflict_cols)}개 → {sorted(list(conflict_cols))[:5]}...")
                    conflict_found = True

        if not conflict_found:
            print("  ✅ 충돌 컬럼 없음")

        return key_info, common_all

    # ================================================================
    # PHASE 2: 1:N 관계 해소 (학교 단위 집계)
    # ================================================================

    def aggregate_to_school_level(df, key_col, name):
        """
        학교코드 기준 중복 행을 학교 단위로 집계
        - 수치형: mean (결측 플래그는 max)
        - 범주형: 최빈값
        """

        if key_col is None:
            print(f"  [{name}] ⚠️ 키 컬럼 없음 → 집계 스킵")
            return df

        n_unique = df[key_col].nunique()
        if n_unique == 0:
            print(f"  [{name}] ⚠️ 유효한 키 값 없음 → 집계 스킵")
            return df

        if n_unique == len(df):
            print(f"  [{name}] 이미 학교 단위 (1:1) → 집계 불필요")
            return df

        n_before = len(df)
        dup_ratio = n_before / n_unique
        print(f"  [{name}] 중복 비율 {dup_ratio:.2f} → 학교 단위 집계 수행")

        # 기본 식별 컬럼 (group key로 유지)
        id_cols = [c for c in ['시도교육청', '교육지원청', '지역', key_col, '학교명',
                                '학교급', '학교급코드', '설립구분', '시도', '시군구'] if c in df.columns]

        # 수치형 vs 범주형 분류
        num_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in id_cols and c not in num_cols]

        # 집계 딕셔너리 구성
        agg_dict = {}
        for col in num_cols:
            if col.endswith('_결측'):
                agg_dict[col] = 'max'  # 하나라도 결측이었으면 1
            else:
                agg_dict[col] = 'mean'

        # 범주형은 안전한 최빈값 함수 사용
        def safe_mode(x):
            m = x.mode()
            return m.iloc[0] if len(m) > 0 else np.nan

        for col in cat_cols:
            agg_dict[col] = safe_mode

        # 식별 컬럼은 first
        id_agg = {col: 'first' for col in id_cols if col != key_col}

        full_agg = {**id_agg, **agg_dict}

        df_agg = df.groupby(key_col, as_index=False).agg(full_agg)
        print(f"    집계 결과: {n_before}행 → {len(df_agg)}행 ({len(df_agg)}개 학교)")

        return df_agg

    # ================================================================
    # PHASE 3: 접두어 부여 및 안전한 병합
    # ================================================================

    def add_prefix_to_features(df, prefix, key_col):
        """기본 식별 컬럼 제외하고 분석 컬럼에 접두어 부여 (병합 시 충돌 방지)"""

        base_cols = {'시도교육청', '교육지원청', '지역', key_col, '학교명',
                    '학교급', '학교급코드', '설립구분', '시도', '시군구',
                    '예결산구분', '세입세출구분'}

        rename_map = {}
        for col in df.columns:
            if col not in base_cols and not col.startswith(prefix + '_'):
                rename_map[col] = f"{prefix}_{col}"

        return df.rename(columns=rename_map)

    def safe_merge_all(dataset_dict, key_col='학교코드', how='inner'):
        """
        모든 데이터셋을 학교코드 기준으로 순차 병합
        - inner: 모든 데이터가 있는 학교만 (보수적)
        - outer: 가능한 모든 학교 포함 (결측 허용)
        """

        print("\n" + "=" * 10)
        print("PHASE 3: 안전한 순차 병합")
        print("=" * 10)

        merge_on = [key_col]
        base_id_cols = ['시도교육청', '교육지원청', '지역', '학교명',
                        '학교급', '학교급코드', '설립구분', '시도', '시군구']

        dfs = list(dataset_dict.items())

        if len(dfs) == 0:
            print("  ⚠️ 병합할 데이터셋이 없습니다.")
            return pd.DataFrame()

        merged = dfs[0][1].copy()
        print(f"  기준 데이터: {dfs[0][0]} ({len(merged)}행)")

        for name, df in dfs[1:]:
            # 병합 대상에서 중복 식별 컬럼 제거 (키 제외)
            drop_cols = [c for c in base_id_cols if c in df.columns and c in merged.columns]
            df_clean = df.drop(columns=drop_cols, errors='ignore')

            before = len(merged)
            merged = merged.merge(df_clean, on=merge_on, how=how, suffixes=('', f'_dup_{name}'))

            # 혹시 남은 _dup 컬럼 제거
            dup_cols = [c for c in merged.columns if '_dup_' in c]
            if dup_cols:
                print(f"    ⚠️ 중복 컬럼 {len(dup_cols)}개 자동 제거: {dup_cols[:3]}...")
                merged.drop(columns=dup_cols, inplace=True)

            print(f"  + {name:<20} → {before:>6}행 → {len(merged):>6}행 (how={how})")

        print(f"\n  ✅ 최종 병합 결과: {merged.shape[0]}행 * {merged.shape[1]}열")
        return merged

    # ================================================================
    # PHASE 4: 통합 파생 변수 (크로스 데이터셋)
    # ================================================================

    def create_cross_features(df):
        """서로 다른 데이터셋의 컬럼을 결합한 파생 변수 생성"""

        print("\n" + "=" * 10)
        print("PHASE 4: 크로스 데이터셋 파생 변수 생성")
        print("=" * 10)

        created = []

        # ★ 헬퍼: 컬럼 존재 + 0→NaN 안전 나눗셈
        def safe_div(numerator_col, denominator_col, result_name, decimals=2):
            if numerator_col in df.columns and denominator_col in df.columns:
                denom = df[denominator_col].replace(0, np.nan)
                df[result_name] = (df[numerator_col] / denom).round(decimals)
                created.append(result_name)
                return True
            return False

        # ★ 헬퍼: 컬럼 안전 접근
        def col_or_zero(col_name):
            return df[col_name] if col_name in df.columns else 0

        # ----- 재정 * 학생 수 -----
        safe_div('세입_수입_총계', '급식_전체학생수', '학생1인당_수입', 0)
        safe_div('세출_세출_총계', '급식_전체학생수', '학생1인당_세출', 0)

        # ----- 재정 * 공간 -----
        safe_div('세출_세출_총계', '공간_전체공간_합계', '공간1실당_세출', 0)

        # ----- 공간 * 학생 수 -----
        safe_div('공간_전체공간_합계', '급식_전체학생수', '학생1인당_전체공간', 4)
        safe_div('공간_교실_합계',    '급식_전체학생수', '학생1인당_교실', 4)
        safe_div('공간_특별실_합계',  '급식_전체학생수', '학생1인당_특별실', 4)
        safe_div('공간_화장실_합계',  '급식_전체학생수', '학생1인당_화장실', 4)
        safe_div('공간_학생복지공간_합계', '급식_전체학생수', '학생1인당_복지공간', 4)

        # ----- 급식 인력 * 재정 -----
        safe_div('세출_세출_학생복지_교육격차해소', '급식_총급식인력', '급식인력1인당_복지세출', 0)

        # ----- 보건실 이용밀도 -----
        safe_div('보건실_보건실_연간이용건수', '공간_보건실', '보건실_이용밀도', 2)

        # ----- 방과후: 대표 지표 1개 + 미운영 플래그 -----
        safe_div('방과후_총_프로그램수', '급식_전체학생수', '학생백명당_방과후프로그램_raw', 4)
        if '학생백명당_방과후프로그램_raw' in df.columns:
            df['학생백명당_방과후프로그램'] = (df['학생백명당_방과후프로그램_raw'] * 100).round(2)
            df.drop(columns=['학생백명당_방과후프로그램_raw'], inplace=True)
            created.append('학생백명당_방과후프로그램')

        if '방과후_총_프로그램수' in df.columns:
            df['방과후_미운영'] = (df['방과후_총_프로그램수'].fillna(0) == 0).astype(int)
            created.append('방과후_미운영')


        # ----- 교육 불평등 복합 지수 -----
        inequality_components = []
        component_names = []

        candidates = {
            '학생1인당_수입':           ('재정',       False),  # 높을수록 좋음
            '학생1인당_전체공간':        ('공간',       False),
            '급식_인력당학생수':         ('급식인력',    True),   # 높을수록 나쁨
            '학생1인당_화장실':         ('위생',       False),
            '학생백명당_방과후프로그램':  ('방과후기회',  False),  # 높을수록 좋음
        }


        for col, (label, reverse) in candidates.items():
            if col in df.columns and df[col].std() > 0:
                z = (df[col] - df[col].mean()) / df[col].std()
                if not reverse:
                    z = -z
                inequality_components.append(z)
                component_names.append(label)

        if len(inequality_components) >= 3:
            composite = pd.concat(inequality_components, axis=1).mean(axis=1)
            df['교육불평등_복합지수'] = composite.round(4)

            # 분위수 기반 등급 (동일값 처리 위해 duplicates='drop')
            try:
                df['교육불평등_등급'] = pd.cut(
                    df['교육불평등_복합지수'],
                    bins=[-np.inf,
                        df['교육불평등_복합지수'].quantile(0.2),
                        df['교육불평등_복합지수'].quantile(0.4),
                        df['교육불평등_복합지수'].quantile(0.6),
                        df['교육불평등_복합지수'].quantile(0.8),
                        np.inf],
                    labels=['매우양호', '양호', '보통', '취약', '매우취약'],
                    duplicates='drop'
                )
            except ValueError:
                # 분위수 중복 시 qcut으로 대체
                df['교육불평등_등급'] = pd.qcut(
                    df['교육불평등_복합지수'], q=5,
                    labels=['매우양호', '양호', '보통', '취약', '매우취약'],
                    duplicates='drop'
                )

            created += ['교육불평등_복합지수', '교육불평등_등급']
            print(f"  ✅ 교육불평등 복합지수 구성요소: {component_names}")
        else:
            print(f"  ⚠️ 복합지수 생성 불가 (유효 구성요소 {len(inequality_components)}개, 최소 3개 필요)")
            print(f"     확인된 구성요소: {component_names}")

        # inf 처리
        df = df.replace([np.inf, -np.inf], np.nan)

        print(f"  ✅ 생성된 크로스 파생 변수 ({len(created)}개):")
        for i, c in enumerate(created):
            print(f"     {i+1:>2}. {c}")

        return df

    # ================================================================
    # PHASE 5: 최종 품질 검증
    # ================================================================

    def final_quality_check(df):
        """최종 병합 데이터의 품질 검증"""

        print("\n" + "=" * 10)
        print("PHASE 5: 최종 품질 검증")
        print("=" * 10)

        print(f"\n  Shape: {df.shape}")
        print(f"  메모리: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # 결측률 높은 컬럼
        missing_pct = (df.isnull().sum()/ len(df) * 100).sort_values(ascending=False)  # 결측률 높은 순 정렬

        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            print(f"\n  ⚠️ 결측률 50% 이상 컬럼 ({len(high_missing)}개):")
            for col, pct in high_missing.head(15).items():
                print(f"    {col}: {pct:.1f}%")
        else:
            print("\n  ✅ 결측률 50% 이상 컬럼 없음")

        # 수치형/범주형 요약
        num_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n  수치형 컬럼: {len(num_cols)}개")
        print(f"  범주형 컬럼: {len(df.columns) - len(num_cols)}개")

        # 학교급 분포
        for label_col in ['학교급', '학교급코드']:
            if label_col in df.columns:
                print(f"\n  {label_col} 분포:")
                for val, cnt in df[label_col].value_counts().items():
                    print(f"    {val}: {cnt}")
                break

        # 시도교육청 분포
        if '시도교육청' in df.columns:
            print(f"\n  시도교육청 분포 (상위 5):")
            for val, cnt in df['시도교육청'].value_counts().head(5).items():
                print(f"    {val}: {cnt}")

        # 크로스 변수 유효율
        cross_cols = ['학생1인당_수입', '학생1인당_전체공간', '학생1인당_화장실',
                    '방과후_참여율', '재정수지율', '교육불평등_복합지수']
        available_cross = [c for c in cross_cols if c in df.columns]
        if available_cross:
            print(f"\n  크로스 변수 유효율:")
            for col in available_cross:
                valid_pct = (df[col].notna().sum() / len(df) * 100)
                print(f"    {col}: {valid_pct:.1f}%")

        return df

    # ================================================================
    # 실행 파이프라인
    # ================================================================

    # 0. 전처리 완료된 데이터셋 모음
    datasets = {
        '실내공기질': df_Campus_Hygiene,
        '방과후':     df_Afterschool_Program,
        '보건실':     df_Campus_HealthOffice,
        '세입':       df_Campus_Income,
        '세출':       df_Campus_Outcome,
        '급식':       df_Campus_Lunch,
        '공간':       df_Campus_Space,
    }

    # 키 컬럼 통일 (정보공시 학교코드 → 학교코드)
    for name, df in datasets.items():
        if '정보공시 학교코드' in df.columns and '학교코드' not in df.columns:
            df.rename(columns={'정보공시 학교코드': '학교코드'}, inplace=True)
            print(f"  [{name}] '정보공시 학교코드' → '학교코드' 리네임")

    # PHASE 1: 진단
    key_info, common_schools = diagnose_datasets(datasets)

    # PHASE 2: 학교 단위 집계
    print("\n" + "=" * 10)
    print("PHASE 2: 학교 단위 집계 (1:N → 1:1)")
    print("=" * 10)

    aggregated = {}
    for name, df in datasets.items():
        kc = key_info[name]['key_col']
        aggregated[name] = aggregate_to_school_level(df, kc, name)

    # PHASE 3: 접두어 + 병합
    prefixed = {}
    for name, df in aggregated.items():
        prefixed[name] = add_prefix_to_features(df, name, '학교코드')

    # 병합 순서: 행 수가 많은 것부터 (기준 데이터)
    sorted_names = sorted(prefixed.keys(), key=lambda x: len(prefixed[x]), reverse=True)
    sorted_datasets = {name: prefixed[name] for name in sorted_names}

    df_merged = safe_merge_all(sorted_datasets, key_col='학교코드', how='outer')

    # PHASE 4: 크로스 파생 변수
    df_merged = create_cross_features(df_merged)

    # PHASE 5: 품질 검증
    df_final = final_quality_check(df_merged)

    print("\n" + "=" * 10)
    print("✅ 전체 파이프라인 완료")
    print(f"   최종 데이터: {df_final.shape[0]}행 * {df_final.shape[1]}열")
    print(f"   공통 학교 수: {len(common_schools)}개")
    print("=" * 10)
    
    # 예시: 결측 플래그 컬럼 제거 (분석 단계에서만 사용)
    flag_cols = [c for c in df_final.columns if c.endswith('_결측')]
    df_final = df_final.drop(columns=flag_cols)

    # 또는 대시보드에 필요한 컬럼만 선별
    dashboard_cols = [
        # 식별
        '시도교육청', '교육지원청', '지역', '학교코드', '학교명', '학교급', '설립구분',

        # 복합지수
        '교육불평등_복합지수', '교육불평등_등급',

        # 급식/학생 (접두어 반영)
        '급식_전체학생수', '급식_학교규모', '급식_급식비율', '급식_급식참여등급',
        '급식_영양교사배치', '급식_인력적정성', '급식_인력당학생수',

        # 재정 (크로스 변수)
        '학생1인당_수입', '학생1인당_세출',

        # 공간 (접두어 반영)
        '학생1인당_전체공간', '학생1인당_교실', '학생1인당_화장실',

        # 위생 (접두어 반영)
        '실내공기질_공기질_등급',

        # 보건실 (접두어 반영)
        '보건실_보건실_보유', '보건실_이용밀도',

        # 방과후
        '학생백명당_방과후프로그램', '방과후_미운영',
    ]
    existing_dashboard = [c for c in dashboard_cols if c in df_final.columns]
    df_dashboard = df_final[existing_dashboard]

    return df_dashboard