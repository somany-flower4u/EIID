# 교육 불평등 지수 대시보드 (EIID)

교육부 **「공공데이터 기반 교육 불평등 지수 대시보드」** 과제 참가 코드입니다. 학교 단위 공공데이터 7종을 학교코드로 합치고, 학교별 교육 불평등 복합지수와 5단계 등급을 만든 뒤, 등급을 예측하는 모델과 그 해석(SHAP)을 Streamlit 대시보드로 보여 줍니다.

> English summary: a Streamlit dashboard for the Korean Ministry of Education public-data task. It merges seven school-level public datasets by school code, builds a composite education inequality index with five grades, trains RandomForest and XGBoost classifiers, and explains predictions with SHAP. The dashboard has data search, school diagnosis, SHAP, policy simulator and model performance pages.

## 지수 정의

`processed_data_merge.py`가 병합 단계에서 지수와 등급을 만듭니다.

- **복합지수(`교육불평등_복합지수`):** 다섯 지표를 z-점수로 바꿔 평균합니다. 값이 클수록 불평등이 큽니다.
  - 학생1인당_수입 (높을수록 양호)
  - 학생1인당_전체공간 (높을수록 양호)
  - 학생1인당_화장실 (높을수록 양호)
  - 학생백명당_방과후프로그램 (높을수록 양호)
  - 급식_인력당학생수 (높을수록 불리)
- **등급(`교육불평등_등급`):** 복합지수를 5분위로 나눠 `매우양호`, `양호`, `보통`, `취약`, `매우취약`으로 부릅니다. 병합 결과 `data/merge_data.csv`에서는 등급당 477개 학교입니다.

## 데이터 처리와 모델

1. **전처리** (`data_preprocessing.py`): CSV 7종을 각각 정리하고 파생 변수를 만듭니다.
2. **병합** (`processed_data_merge.py`, `main.py`): 학교코드 기준으로 합칩니다. 한 학교에 행이 여러 개이면 수치형은 평균, 범주형은 최빈값으로 모읍니다. 결과는 `data/merge_data.csv`입니다. 2,391행입니다.
3. **탐색, 학습, 해석** (`SHAP.py`): 이름과 달리 이 스크립트가 EDA, 모델 학습, SHAP 분석, 산출물 저장을 모두 합니다.
   - 입력: 수치 9개(급식 전체학생수, 급식비율, 인력당 학생수, 학생1인당 수입·전체공간·교실·화장실, 학생백명당 방과후프로그램, 방과후 미운영)와 범주 6개(학교급, 설립구분, 급식 학교규모, 급식참여등급, 영양교사배치, 인력적정성)
   - 모델: RandomForest(300 trees, max_depth 10)와 XGBoost(300 trees, max_depth 6, learning_rate 0.1)를 학습하고 가중 F1이 높은 쪽을 최적 모델로 고릅니다.
   - 분할: 학습 80%, 테스트 20%(계층 분할, 시드 42), 5-fold 교차검증도 함께 계산합니다.
   - 저장: `models/`에 모델, 라벨 인코더, 피처 목록, 성능 지표, 분할 인덱스, SHAP 값을 저장합니다. EDA 그림은 `plot/`, 상위 5개 학교의 SHAP 워터폴은 `Waterfall/`에 있습니다.

### 저장된 성능 (`models/metrics.pkl`)

| 모델 | 학습 Accuracy | 테스트 Accuracy | 테스트 F1(가중) |
|---|---:|---:|---:|
| RandomForest | 0.9817 | 0.8889 | 0.8893 |
| XGBoost | 1.0000 | 0.9245 | 0.9241 |

복합지수를 이루는 다섯 지표가 모두 모델 입력에도 들어 있습니다. 그래서 이 성능은 등급을 만드는 규칙을 모델이 얼마나 복원하는지를 보여 주는 값으로 읽어야 합니다.

## 대시보드 화면

`streamlit run app.py`로 실행합니다. 메인 화면에서 아래 페이지로 이동합니다.

| 페이지 | 내용 |
|---|---|
| `pages/1_Data_Search.py` 데이터 탐색 | 기술통계, 결측치, 변수별 분포, 상관관계 히트맵, 시도교육청별 레이더 차트를 봅니다. |
| `pages/2_School_Diagnosis.py` 학교별 진단 | 시도교육청과 학교급으로 학교를 골라 전국, 동일 지역과 지표를 비교합니다. |
| `pages/3_SHAP.py` SHAP 분석 | 전역 변수 중요도와 학교별 SHAP 기여도를 봅니다. |
| `pages/4_Simulator.py` 정책 시뮬레이션 | 재정 확대, 방과후 프로그램 확충, 시설 개선 시나리오를 정해 현재와 시뮬레이션 결과를 비교합니다. |
| `pages/5_Model_Performance.py` 모델 성능 | 모델별 성능, 혼동 행렬, 클래스별 성능을 봅니다. |

## 폴더 구조

```
app.py                 대시보드 메인
main.py                전처리와 병합 실행
data_preprocessing.py  데이터셋별 전처리 함수
processed_data_merge.py  병합, 복합지수와 등급 생성
SHAP.py                EDA, 모델 학습, SHAP, 산출물 저장
test.ipynb             실험 노트북
pages/                 Streamlit 페이지 5개
utils/data_loader.py   데이터, 모델 로딩과 캐시
data/                  원본 CSV 7종과 merge_data.csv
models/                학습된 모델, 인코더, 지표, SHAP 값 (pkl)
plot/                  EDA, 성능, 변수 중요도, SHAP beeswarm 그림
Waterfall/             상위 5개 학교의 SHAP 워터폴 그림
```

## 설치와 실행

```bash
pip install -r requirements.txt
```

대시보드만 보려면 저장된 `data/merge_data.csv`와 `models/`로 바로 실행됩니다.

```bash
streamlit run app.py
```

처음부터 다시 만들려면 아래 순서로 실행합니다. 저장소 루트에서 실행하고, `models/`, `plot/`, `Waterfall/` 폴더가 있어야 합니다.

```bash
python main.py     # 전처리와 병합 -> data/merge_data.csv
python SHAP.py     # 학습, SHAP, models/ 산출물 저장
```

`models/*.pkl`은 joblib으로 저장했습니다. 출처를 모르는 pkl은 열지 마세요. 같은 버전의 scikit-learn, XGBoost, SHAP에서 여는 것이 안전합니다.

## 데이터

- **출처:** 교육부 관련 학교 단위 공공데이터입니다. 저장소에는 CSV 7종이 들어 있습니다.
  - 학교 위생상태(2023): `2023_campus_hygiene.csv`
  - 급식, 학교 수입, 교육공간, 보건실, 방과후학교, 학생 성과(2025): `2025_campus_lunch.csv`, `2025_campus_income.csv`, `2025_campus_space.csv`, `2025_campus_healthoffice.csv`, `2025_afterschool_program.csv`, `2025_campus_outcome.csv`
- 학교코드는 원본의 `정보공시 학교코드` 열을 씁니다.
- 병합 데이터와 학습된 모델도 저장소에 포함되어 있습니다.
- 데이터의 이용 조건은 원 제공처의 조건(예: 공공누리)을 따릅니다. 개별 파일의 제공처와 이용 유형은 저장소에 기록되어 있지 않아 확인하지 못했습니다.

## 라이선스

코드는 [MIT License](LICENSE)를 따릅니다. 저장소에 포함된 공공데이터는 원 제공처의 이용 조건을 따릅니다.
