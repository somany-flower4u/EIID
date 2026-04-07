# 📊 공공데이터 기반 교육 불평등 지수 대시보드

**Education Inequality Index Dashboard Based on Public Data**

---

## 📌 프로젝트 개요

이 프로젝트는 공공데이터를 활용하여 **학교별 교육 불평등 지수를 분석하고 시각화**하는 대시보드입니다. 
7개의 다양한 공공데이터셋(위생상태, 급식, 소득, 교육공간, 보건실, 방과후학교, 학생성과)을 수집하여 통합 분석하며, 
머신러닝 모델을 통해 불평등 요인을 파악하고 SHAP을 이용한 모델 해석성을 제공합니다.

---

## ✨ 주요 기능

- **📈 다중 데이터 통합 분석**: 7개 공공데이터셋 자동 수집 및 통합
- **🎯 데이터 전처리 자동화**: 이상치 탐지, 결측치 처리, 정규화 등 자동 수행
- **📊 대시보드 시각화**: Streamlit 기반의 인터랙티브 웹 인터페이스
- **🔍 학교 진단**: 개별 학교의 불평등 지수 분석 및 진단
- **🤖 머신러닝 모델 통합**: 불평등 지수 예측 모델 구현
- **📋 SHAP 해석성**: 모델 의사결정 과정 시각화 및 설명
- **⚙️ 시뮬레이터**: 정책 변수 변화에 따른 영향도 분석

---

## 📁 프로젝트 구조

```
Public_Data-Based_Education_Inequality_Index_Dashboard/
├── 📄 app.py                          # Streamlit 메인 대시보드
├── 📄 main.py                         # 데이터 수집 및 병합 실행 스크립트
├── 📄 data_preprocessing.py           # 개별 데이터셋 전처리 함수 정의
├── 📄 processed_data_merge.py         # 데이터 병합 로직
├── 📄 SHAP.py                         # EDA 및 데이터 탐색
├── 📄 test.ipynb                      # 테스트 및 실험 노트북
│
├── 📁 data/                           # 데이터 저장 디렉토리
│   ├── 2023_campus_hygiene.csv        # 학교 위생상태 데이터
│   ├── 2025_campus_lunch.csv          # 급식 현황 데이터
│   ├── 2025_campus_income.csv         # 학교 수입 데이터
│   ├── 2025_campus_space.csv          # 교육공간 데이터
│   ├── 2025_campus_healthoffice.csv   # 보건실 운영 데이터
│   ├── 2025_afterschool_program.csv   # 방과후학교 데이터
│   ├── 2025_campus_outcome.csv        # 학생 성과 데이터
│   └── merge_data.csv                 # 최종 통합 데이터 (자동 생성)
│
├── 📁 pages/                          # Streamlit 멀티페이지 앱
│   ├── 1_Data_Search.py               # 데이터 검색 및 필터링
│   ├── 2_School_Diagnosis.py          # 학교별 진단 분석
│   ├── 3_SHAP.py                      # SHAP 모델 해석성 분석
│   ├── 4_Simulator.py                 # 정책 시뮬레이터
│   └── 5_Model_Performance.py         # 모델 성능 평가
│
├── 📁 utils/                          # 유틸리티 함수
│   ├── __init__.py
│   └── data_loader.py                 # 데이터 로딩 및 캐싱 함수
│
├── 📁 models/                         # 학습된 머신러닝 모델 저장
├── 📁 Waterfall/                      # Waterfall 차트 관련 파일
├── 📁 plot/                           # 시각화 결과 저장
│
└── 📄 README.md                       # 프로젝트 설명 문서
```

---

## 🔄 데이터 처리 프로세스

### **Phase 1: 개별 데이터 전처리** (`data_preprocessing.py`)

프로젝트의 7개 CSV 데이터셋을 개별적으로 전처리합니다.

| # | 데이터셋 | 파일명 | 주요 처리 항목 |
|---|---------|--------|--------------|
| 1 | 위생상태 | `2023_campus_hygiene.csv` | PM10/PM2.5, CO2, HCHO, 세균(부유/낙하), 라돈, TVOC 측정값 정규화 |
| 2 | 급식 현황 | `2025_campus_lunch.csv` | 급식 비율, 인력당 학생수, 영양교사 배치 정보 |
| 3 | 학교 수입 | `2025_campus_income.csv` | 학생 1인당 교육경상경비 계산 |
| 4 | 교육공간 | `2025_campus_space.csv` | 학생 1인당 교실/화장실/전체 면적 계산 |
| 5 | 보건실 운영 | `2025_campus_healthoffice.csv` | 보건 인력, 의료 장비 현황 집계 |
| 6 | 방과후학교 | `2025_afterschool_program.csv` | 프로그램 수, 참여 학생 비율 |
| 7 | 학생 성과 | `2025_campus_outcome.csv` | 표준화 점수, 대학 진학률 |

**각 전처리 함수의 공통 작업:**
1. **컬럼 리네임**: 통일된 한국어 기준명으로 변환
2. **행 필터링**: 제외 대상 학교, 결측 학교코드 제거
3. **데이터 타입 변환**: 수치형/범주형 데이터 정규화
4. **파생 변수 생성**: 평균, 합계, 비율 등 새로운 변수 계산
5. **결측치 처리**: 지역+학교급별 중앙값으로 대체

---

### **Phase 2: 데이터 병합** (`processed_data_merge.py`)

학교코드를 기준으로 모든 전처리된 데이터를 통합합니다.

```
병합 프로세스 흐름:

├─ PHASE 1: 호환성 진단
│  ├─ 각 데이터셋의 행수, 열수, 고유 학교수 확인
│  ├─ 학교코드 교집합 분석 (모든 데이터셋에 공통 학교 확인)
│  ├─ 학교코드 중복도 검사 (1:N 관계 탐지)
│  └─ 컬럼명 충돌 사전 점검
│
├─ PHASE 2: 1:N 관계 해소
│  ├─ 수치형 변수: 평균(mean) 계산으로 집계
│  ├─ 범주형 변수: 최빈값(mode)으로 결정
│  └─ 결측 플래그: 최댓값(max) - 하나라도 결측값이면 1로 표시
│
└─ PHASE 3: 최종 병합
   └─ 학교코드 기준 outer join → merge_data.csv로 저장
```

---

### **Phase 3: 메인 처리 실행** (`main.py`)

모든 데이터 처리를 자동으로 실행합니다.

**실행 순서:**
1. 각 CSV 파일 독립 로드 → 해당 전처리 함수 적용
2. `merge_data()` 함수 호출로 병합
3. 최종 병합 결과를 `data/merge_data.csv`로 저장

```bash
python main.py
```

---

### **Phase 4: 탐색적 데이터 분석** (`SHAP.py`)

병합된 데이터의 기본 통계 및 시각화를 수행합니다.

- 기본 통계: 행수, 열수, 데이터 타입, 결측치 현황
- 타겟 변수 분포: 교육불평등_등급, 교육불평등_복합지수
- 시각화:
  - 불평등 등급 분포 (카운트)
  - 복합지수 분포 (히스토그램)
  - 학교급별 불평등 지수 (박스플롯)
  - 기타 탐색적 시각화

```bash
python SHAP.py
```

---

## 🚀 설치 및 실행 방법

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/Public_Data-Based_Education_Inequality_Index_Dashboard.git
cd Public_Data-Based_Education_Inequality_Index_Dashboard

# Python 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 2️⃣ 필수 패키지 설치

```bash
pip install -r requirements.txt
```

또는 필수 패키지를 직접 설치:

```bash
pip install streamlit pandas numpy plotly scikit-learn shap matplotlib seaborn scipy
```

### 3️⃣ 데이터 준비

`data/` 디렉토리에 다음 CSV 파일들을 배치합니다:
- `2023_campus_hygiene.csv`
- `2025_campus_lunch.csv`
- `2025_campus_income.csv`
- `2025_campus_space.csv`
- `2025_campus_healthoffice.csv`
- `2025_afterschool_program.csv`
- `2025_campus_outcome.csv`

### 4️⃣ 데이터 처리 및 병합

```bash
python main.py
```

이 명령으로 `data/merge_data.csv`가 자동 생성됩니다.

### 5️⃣ 대시보드 실행

```bash
streamlit run app.py
```

웹 브라우저가 자동으로 열리며, `http://localhost:8501`에서 대시보드를 볼 수 있습니다.

---

## 📖 파일 상세 설명

### **app.py** - 메인 대시보드
- Streamlit 기반의 메인 인터페이스
- 프로젝트 개요, 통계, 주요 시각화 표시
- 사이드바를 통한 네비게이션
- 커스텀 CSS 스타일링 적용

### **pages/1_Data_Search.py** - 데이터 검색
- 통합된 데이터를 필터링 및 검색
- 학교명, 지역, 학교급 등으로 데이터 조회
- 결과를 테이블 형식으로 표시 및 다운로드 기능

### **pages/2_School_Diagnosis.py** - 학교 진단
- 선택한 학교의 상세 분석 수행
- 학교별 불평등 지수 및 세부 지표 표시
- 벤치마크 비교 (전국 평균, 지역 평균 등)
- 개선 권고사항 제시

### **pages/3_SHAP.py** - 모델 해석성
- SHAP(SHapley Additive exPlanations) 분석
- 각 변수의 모델 기여도 시각화
- 개별 학교의 예측에 영향을 미치는 주요 요소 분석
- Waterfall plot, Force plot 등 SHAP 시각화

### **pages/4_Simulator.py** - 정책 시뮬레이터
- 각 지표 값을 조정하여 불평등 지수 변화 시뮬레이션
- 정책 효과 분석 및 예측
- 최적 개선 방안 제시

### **pages/5_Model_Performance.py** - 모델 성능
- 예측 모델의 성능 평가 지표
- 혼동 행렬, ROC 곡선, 정확도 등
- 모델 학습 곡선 및 교차 검증 결과

### **data_preprocessing.py**
8개의 클래스 기반 전처리 함수 포함:
- `Campus_Hygiene_Data_Processing()`: 위생상태 데이터 전처리
- `Campus_Lunch_Data_Processing()`: 급식 데이터 전처리
- `Campus_Income_Data_Processing()`: 수입 데이터 전처리
- `Campus_Space_Data_Processing()`: 교육공간 데이터 전처리
- `Campus_Health_Office_Data_Processing()`: 보건실 데이터 전처리
- `Afterschool_Program_Data_Processing()`: 방과후학교 데이터 전처리
- `Campus_Outcome_Data_Processing()`: 학생 성과 데이터 전처리
- 각 함수는 컬럼 리네임, 필터링, 파생 변수 생성 등 수행

### **processed_data_merge.py**
- `merge_data()`: 전처리된 7개 데이터셋을 학교코드 기준으로 병합
- 호환성 진단, 중복값 처리, 최종 병합 로직 포함

### **utils/data_loader.py**
- `init_session()`: Streamlit 세션 초기화 및 데이터 로딩
- 데이터 캐싱으로 성능 최적화

### **SHAP.py**
- 병합된 데이터의 탐색적 데이터 분석 수행
- 기본 통계, 결측치 분석, 분포 시각화

---

## 🛠 기술 스택

| 분야 | 기술 |
|------|------|
| **웹 프레임워크** | Streamlit |
| **데이터 처리** | Pandas, NumPy |
| **시각화** | Plotly, Matplotlib, Seaborn |
| **머신러닝** | Scikit-learn |
| **모델 해석성** | SHAP |
| **언어** | Python 3.8+ |

---

## 📊 주요 지표 정의

### 교육불평등_등급
- **S등급**: 매우 우수 (불평등 매우 적음)
- **A등급**: 우수 (불평등 적음)
- **B등급**: 보통
- **C등급**: 미흡 (불평등 많음)
- **D등급**: 매우 미흡 (불평등 심각)

### 교육불평등_복합지수
- 7개 지표를 종합하여 계산한 0~100 점수
- 점수가 높을수록 불평등이 심함

---

## 💡 사용 예시

### 1. 기본 데이터 처리
```bash
# 데이터 전처리 및 병합
python main.py

# 탐색적 분석 수행
python SHAP.py
```

### 2. 대시보드 실행
```bash
streamlit run app.py
```

### 3. 특정 페이지 접근 (앱 실행 후)
- **Data Search**: 학교 데이터 검색 및 조회
- **School Diagnosis**: 특정 학교 진단 분석
- **SHAP**: 모델 해석성 분석
- **Simulator**: 정책 시나리오 분석
- **Model Performance**: 예측 모델 성능 확인

---

## 📋 향후 개선 계획

- [ ] 실시간 데이터 업데이트 자동화
- [ ] 추가 머신러닝 모델 통합 (LightGBM, XGBoost 등)
- [ ] 예측 정확도 향상을 위한 하이퍼파라미터 튜닝
- [ ] 시계열 분석 추가
- [ ] 모바일 앱 개발
- [ ] 데이터 API 서빙

---

**마지막 업데이트**: 2026년 4월 7일