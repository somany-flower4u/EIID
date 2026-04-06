# Public_Data-Based_Education_Inequality_Index_Dashboard
공공데이터 기반 교육 불평등 지수 대시보드

**교육 불평등 지수 대시보드 - 데이터 처리 및 분석 절차**
**Phase 1: 데이터 수집 & 개별 전처리 (data_preprocessing.py)**
프로젝트에서는 7개의 독립적인 CSV 데이터셋을 수집하여 개별적으로 전처리합니다.

*데이터셋 파일명 주요 처리*
1. 학교 위생상태     2023_campus_hygiene.csv	      PM10/PM2.5, CO2, HCHO, 세균, 라돈 등 검사 결과 정규화
2. 급식 현황        2025_campus_lunch.csv	        급식 비율, 인력당 학생수, 영양 교사 배치 여부
3. 학교 수입        2025_campus_income.csv	        학생1인당 수입 (교육현경비)
4. 교육공간         2025_campus_space.csv	        학생1인당 교실, 화장실, 전체공간 면적
5. 보건실 운영      2025_campus_healthoffice.csv	  보건 인력, 의료 장비 현황
6. 방과후학교       2025_afterschool_program.csv	  프로그램 수, 참여 학생 수
7. 학생 성과        2025_campus_outcome.csv	      표준화 점수, 대입 실적

*각 전처리 함수의 공통 작업*
1. 컬럼 리네임:     통일된 한국어 기준명으로 변환
2. 행 필터링:       제외 학교, 결측 학교코드 제거
3. 타입 변환:       수치형/범주형 데이터 정규화
4. 파생 변수 생성:  평균, 합계, 비율 등 계산
5. 결측치 처리:     지역+학교급별 중앙값 대체

**Phase 2: 데이터 병합 (processed_data_merge.py)**
*학교코드 기준 병합 (학교별 1:1 관계)*
├─ PHASE 1: 병합 호환성 진단  
│  ├─ 각 데이터셋 행수, 열수, 고유학교수 확인  
│  ├─ 학교코드 교집합 분석 (모든 데이터셋이 공통으로 가진 학교 확인)  
│  ├─ 학교코드 중복도 검사 (1:N 관계 탐지)  
│  └─ 컬럼명 충돌 사전 점검  
├─ PHASE 2: 1:N 관계 해소 (중복 행 집계)  
│  ├─ 수치형 변수: 평균(mean) 계산   
│  ├─ 범주형 변수: 최빈값(mode) 계산  
│  └─ 결측 플래그: 최댓값(max) - 하나라도 결측이면 1  
└─ PHASE 3: 최종 병합  
   └─ 학교코드 기준 outer join → merge_data.csv 저장  

**Phase 3: 메인 처리 실행 (main.py)**
*실행 순서*
1. 각 CSV 독립 로드 → 해당 전처리 함수 적용
2. 병합 함수 호출
3. 최종 병합 데이터 저장 (merge_data.csv)

**Phase 4: 탐색적 데이터 분석 (EDA) (SHAP.py)**
1. 기본 통계: 행수, 열수, 데이터 타입, 결측치
2. 타겟 변수 분포: 교육불평등_등급, 교육불평등_복합지수
3. 시각화 탐색 (6가지)  
  ├─ 불평등 등급 분포 (카운트)  
  ├─ 복합지수 분포 (히스토그램)  
  ├─ 학교급별 불평등 지수 (박스플롯)  
  ├─ 학생1인당 수입 vs 지수 (산점도)  
  ├─ 방과후 프로그램 vs 지수 (산점도)  
  └─ 시도교육청별 평균 지수 (막대그래프)  
4. 상관관계 분석: 주요 변수 간 피어슨 상관계수  

**Phase 5: Streamlit 대시보드 (app.py, pages)**
*메인 페이지 (app.py)*  
  ├─ 메트릭 카드: 전체 학교 수, 변수 개수  
  ├─ 다중 필터: 시도교육청, 학교급, 불평등 등급  
  └─ 대시보드 시각화  

*상세 분석 페이지 (pages)*
      페이지	                      기능
1_Data_Search.py	      학교별 상세 데이터 검색 및 조회
2_School_Diagnosis.py  	학교 진단 결과 시각화
3_SHAP.py	              모델 해석가능성 분석 (SHAP value)
4_Simulator.py	        시뮬레이터: 변수 값 조정 시 불평등 등급 예측
5_Model_Performance.py	머신러닝 모델 성능 지표

*피처 구성*
수치형 피처 (9개): 급식 비율, 학생1인당 수입, 교실 면적 등
범주형 피처 (4개): 학교급, 설립구분, 급식 규모 등

**데이터 흐름 다이어그램**
CSV 데이터 (7개)
    ↓
개별 전처리 함수들  
├─ Campus_Hygiene_Data_Processing  
├─ Afterschool_Program_Data_Processing  
├─ Campus_Health_Office_Data_Processing  
├─ Campus_Income_Data_Processing  
├─ Campus_Outcome_Data_Processing  
├─ Campus_Lunch_Data_Processing  
└─ Campus_Space_Data_Processing  
    ↓  
merge_data() 함수  
├─ 호환성 진단  
├─ 1:N 관계 해소  
└─ 학교코드 기준 병합  
    ↓
merge_data.csv (최종 병합 데이터)  
    ↓
┌─────────────┬────────────────┬─────────────────┐  
│             │                │                 │  
↓             ↓                ↓                 ↓  
EDA      Streamlit       모델 학습          시뮬레이터  
분석      대시보드      (Model 저장)        예측  

주요 특징
1. 자동화된 결측치 처리: 지역×학교급 중앙값으로 스마트하게 대체
2. 호환성 검증: 병합 전 7개 데이터셋의 키 매칭 및 충돌 사전 진단
3. 다단계 집계: 중복된 행을 학교 단위로 통합 (수치형→평균, 범주형→최빈값)
4. 대시보드 제공: Streamlit으로 대화형 분석 인터페이스 구현

이 절차를 통해 교육 불평등 지수를 정량화하고 모델 기반 예측 및 해석이 가능합니다.

