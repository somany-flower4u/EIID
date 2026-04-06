# ============================================
# Phase 1: 데이터 탐색 (EDA)
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 데이터 로드 ──
df = pd.read_csv('data/merge_data.csv')

print("=" * 10)
print(f"데이터 크기: {df.shape[0]}개 학교, {df.shape[1]}개 변수")
print("=" * 10)
print()

# ── 2. 기본 정보 ──
print("▶ 컬럼 목록 및 데이터 타입:")
print(df.dtypes)
print()

print("▶ 결측치 현황:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'결측수': missing, '비율(%)': missing_pct})
print(missing_df[missing_df['결측수'] > 0])
print()

# ── 3. 타겟 변수 분포 ──
print("▶ 교육불평등_등급 분포:")
print(df['교육불평등_등급'].value_counts())
print()

print("▶ 교육불평등_복합지수 기초통계:")
print(df['교육불평등_복합지수'].describe())

# ============================================
# Phase 1-2: 시각화 탐색
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('교육 불평등 데이터 탐색', fontsize=16, fontweight='bold')

# (1) 불평등 등급 분포
grade_order = df['교육불평등_등급'].value_counts().index.tolist()
sns.countplot(data=df, x='교육불평등_등급', order=grade_order, 
              palette='RdYlGn_r', ax=axes[0, 0])
axes[0, 0].set_title('교육불평등 등급 분포')
axes[0, 0].set_xlabel('')

# (2) 복합지수 히스토그램
sns.histplot(data=df, x='교육불평등_복합지수', bins=30, 
             kde=True, color='steelblue', ax=axes[0, 1])
axes[0, 1].set_title('교육불평등 복합지수 분포')

# (3) 학교급별 복합지수
sns.boxplot(data=df, x='학교급', y='교육불평등_복합지수', 
            palette='Set2', ax=axes[0, 2])
axes[0, 2].set_title('학교급별 불평등 지수')

# (4) 학생1인당 수입 vs 복합지수
sns.scatterplot(data=df, x='학생1인당_수입', y='교육불평등_복합지수',
                hue='교육불평등_등급', palette='RdYlGn_r', 
                alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title('학생1인당 수입 vs 불평등 지수')

# (5) 방과후 프로그램 vs 복합지수
sns.scatterplot(data=df, x='학생백명당_방과후프로그램', y='교육불평등_복합지수',
                hue='교육불평등_등급', palette='RdYlGn_r', 
                alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('방과후 프로그램 vs 불평등 지수')

# (6) 시도교육청별 평균 복합지수
region_mean = df.groupby('시도교육청')['교육불평등_복합지수'].mean().sort_values(ascending=False)
region_mean.plot(kind='barh', color='coral', ax=axes[1, 2])
axes[1, 2].set_title('시도교육청별 평균 불평등 지수')
axes[1, 2].set_xlabel('평균 복합지수')

plt.tight_layout()
plt.savefig('01_EDA_탐색.png', dpi=150, bbox_inches='tight')
#plt.show()
print("✅ 탐색 시각화 저장 완료: 01_EDA_탐색.png")
# ============================================
# Phase 1-3: 상관관계 분석
# ============================================

# 수치형 변수만 추출
numeric_cols = [
    '교육불평등_복합지수', '급식_전체학생수', '급식_급식비율',
    '급식_인력당학생수', '학생1인당_수입', '학생1인당_전체공간',
    '학생1인당_교실', '학생1인당_화장실', '학생백명당_방과후프로그램',
    '방과후_미운영'
]

# 실제 데이터에 존재하는 컬럼만 필터링
numeric_cols = [col for col in numeric_cols if col in df.columns]

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('변수 간 상관관계 히트맵', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('02_상관관계_히트맵.png', dpi=150, bbox_inches='tight')
#plt.show()
print("✅ 상관관계 히트맵 저장 완료: 02_상관관계_히트맵.png")

# 복합지수와의 상관관계 정렬 출력
print("\n▶ 교육불평등_복합지수와의 상관관계 (절대값 기준 정렬):")
corr_with_target = corr_matrix['교육불평등_복합지수'].drop('교육불평등_복합지수')
print(corr_with_target.abs().sort_values(ascending=False).to_string())

# ============================================
# Phase 2: 분류 모델 학습 (수정본)
# ============================================
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# ── XGBoost 임포트 시도 ──
from xgboost import XGBClassifier

# ── 1. 피처/타겟 정의 ──
feature_cols = [
    '급식_전체학생수', '급식_급식비율', '급식_인력당학생수',
    '학생1인당_수입', '학생1인당_전체공간', '학생1인당_교실',
    '학생1인당_화장실', '학생백명당_방과후프로그램', '방과후_미운영'
]

categorical_features = ['학교급', '설립구분', '급식_학교규모',
                        '급식_급식참여등급', '급식_영양교사배치', '급식_인력적정성']

feature_cols = [c for c in feature_cols if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

df_model = df[feature_cols + categorical_features + ['교육불평등_등급']].copy()
df_model = df_model.dropna(subset=['교육불평등_등급'])

df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

X = df_encoded.drop('교육불평등_등급', axis=1)
y = df_encoded['교육불평등_등급']

X = X.fillna(X.median())

# ── 타겟 인코딩 (안전한 타입 처리) ──
le = LabelEncoder()
y_encoded: np.ndarray = np.asarray(le.fit_transform(y))

class_labels: list[str] = np.asarray(le.classes_).tolist()
class_indices: list[int] = np.asarray(le.transform(le.classes_)).tolist()
class_mapping = {label: idx for label, idx in zip(class_labels, class_indices)}
print(f"▶ 등급 클래스: {class_mapping}")
print(f"▶ 피처 수: {X.shape[1]}, 샘플 수: {X.shape[0]}")
print()

# ── 2. 학습/테스트 분할 ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"▶ 학습 세트: {X_train.shape[0]}개, 테스트 세트: {X_test.shape[0]}개")

# ── 3. 모델 학습 ──
models: dict[str, RandomForestClassifier | GradientBoostingClassifier | XGBClassifier] = {}

# (A) RandomForest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=5,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train, y_train)
models['RandomForest'] = rf

# (B) XGBoost 또는 GradientBoosting
# ✅ 수정: HAS_XGB 플래그로 안전하게 분기
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False,
    eval_metric='mlogloss'
    )
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model

# ── 4. 모델 평가 ──
print("\n" + "=" * 10)
print("모델 성능 비교")
print("=" * 10)

best_model_name: str = ""
best_f1: float = 0.0

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average='weighted'))


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='f1_weighted')

    print(f"\n📊 {name}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 (weighted): {f1:.4f}")
    print(f"   CV F1 평균: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

print(f"\n✅ 최적 모델: {best_model_name} (F1={best_f1:.4f})")
best_model = models[best_model_name]
import joblib
joblib.dump(best_model, "models/model.pkl")

# ============================================
# Phase 2-2: 혼동행렬 시각화 (수정본)
# ============================================
y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ✅ 수정: le.classes_.tolist()로 Sequence[str] 타입 보장
cm = confusion_matrix(y_test, y_pred_best)
tick_labels: list[str] = le.classes_.tolist()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=axes[0])
axes[0].set_title(f'{best_model_name} 혼동행렬', fontsize=13)
axes[0].set_xlabel('예측 등급')
axes[0].set_ylabel('실제 등급')

# 분류 리포트
report = classification_report(y_test, y_pred_best,
                                target_names=tick_labels, output_dict=True)
report_df = pd.DataFrame(report).T
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

report_df[['precision', 'recall', 'f1-score']].plot(
    kind='bar', ax=axes[1], colormap='viridis', edgecolor='black'
)
axes[1].set_title('등급별 Precision / Recall / F1', fontsize=13)
axes[1].set_xlabel('등급')
axes[1].set_ylabel('점수')
axes[1].legend(loc='lower right')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('03_모델_성능.png', dpi=150, bbox_inches='tight')
#plt.show()
print("✅ 모델 성능 시각화 저장 완료: 03_모델_성능.png")

# ============================================
# Phase 3: SHAP 분석 (최종 수정)
# ============================================
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap_arr = np.array(shap_values)
print(f"▶ shap_values shape: {shap_arr.shape}")

# ── 차원 구조 판별 ──
# XGBoost 최신: (n_samples, n_features, n_classes) → 3차원
# 구버전/RF:    리스트 of (n_samples, n_features) → 변환 시 (n_classes, n_samples, n_features)

if shap_arr.ndim == 3:
    n_samples, n_features, n_classes = shap_arr.shape

    if n_samples == X_test.shape[0] and n_features == X_test.shape[1]:
        # (샘플, 피처, 클래스) 구조
        print(f"▶ 구조: (샘플={n_samples}, 피처={n_features}, 클래스={n_classes})")

        # 전체 클래스 |SHAP| 평균 → (n_features,) 1차원
        shap_mean = np.abs(shap_arr).mean(axis=2).mean(axis=0)

        # Beeswarm용: 특정 클래스 선택
        if '취약' in le.classes_:
            target_class_idx = list(le.classes_).index('취약')
        elif '심각' in le.classes_:
            target_class_idx = list(le.classes_).index('심각')
        else:
            target_class_idx = n_classes - 1

        shap_values_for_plot = shap_arr[:, :, target_class_idx]  # (샘플, 피처)

    else:
        # (클래스, 샘플, 피처) 구조
        print(f"▶ 구조: (클래스={n_samples}, 샘플={n_features}, 피처={n_classes})")
        shap_mean = np.abs(shap_arr).mean(axis=1).mean(axis=0)

        if '취약' in le.classes_:
            target_class_idx = list(le.classes_).index('취약')
        else:
            target_class_idx = 0

        shap_values_for_plot = shap_arr[target_class_idx]

else:
    # 이진 분류: (n_samples, n_features)
    print("▶ 이진 분류 SHAP")
    shap_mean = np.abs(shap_arr).mean(axis=0)
    shap_values_for_plot = shap_arr
    target_class_idx = 0

target_class_name = le.classes_[target_class_idx]
print(f"▶ shap_mean shape: {shap_mean.shape}")
print(f"▶ 분석 대상 클래스: '{target_class_name}' (index={target_class_idx})")
print(f"▶ shap_values_for_plot shape: {shap_values_for_plot.shape}")

# ============================================
# 변수 중요도 시각화
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# (A) Feature Importance (모델 내장)
feat_imp = pd.Series(
    best_model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(15)

feat_imp.plot(kind='barh', color='steelblue', ax=axes[0])
axes[0].set_title(f'{best_model_name} Feature Importance (Top 15)', fontsize=13)
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance')

# (B) SHAP 평균 |SHAP값|
shap_imp = pd.Series(shap_mean, index=X.columns).sort_values(ascending=False).head(15)
shap_imp.plot(kind='barh', color='coral', ax=axes[1])
axes[1].set_title('SHAP 평균 |SHAP값| (Top 15)', fontsize=13)
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean |SHAP value|')

plt.tight_layout()
plt.savefig('04_변수_중요도.png', dpi=150, bbox_inches='tight')
#plt.show()
print("✅ 변수 중요도 시각화 저장 완료: 04_변수_중요도.png")

# ============================================
# SHAP Beeswarm Plot
# ============================================
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_for_plot,   # (n_samples, n_features) 2차원
    X_test,
    max_display=15,
    show=False
)
plt.title(f'SHAP 분석: "{target_class_name}" 등급 예측 요인', fontsize=13)
plt.tight_layout()
plt.savefig('05_SHAP_beeswarm.png', dpi=150, bbox_inches='tight')
#plt.show()
print("✅ SHAP Beeswarm 저장 완료: 05_SHAP_beeswarm.png")


# ============================================
# Phase 3-2: 개별 학교 해석 (예시) - 수정본
# ============================================

top5_idx = df.nlargest(5, '교육불평등_복합지수').index

print("=" * 10)
print("▶ 교육불평등 지수 상위 5개 학교 분석")
print("=" * 10)

for idx in top5_idx:
    if idx not in X.index:
        continue

    school_name = df.loc[idx, '학교명']
    region = df.loc[idx, '지역']
    grade = df.loc[idx, '교육불평등_등급']
    score = df.loc[idx, '교육불평등_복합지수']

    print(f"\n🏫 {school_name} ({region})")
    print(f"   등급: {grade}, 복합지수: {score:.4f}")

    # 해당 학교의 SHAP 값 계산
    school_data = X.loc[[idx]]
    school_shap = explainer.shap_values(school_data)

    # ━━━━ 수정: school_shap → s_vals 변환 ━━━━
    s_arr = np.array(school_shap)
    print(f"   ★ school_shap shape: {s_arr.shape}")

    # 예측 클래스 확인
    pred_label = int(best_model.predict(school_data)[0])

    if s_arr.ndim == 3:
        # (n_samples, n_features, n_classes) 또는 (n_classes, n_samples, n_features)
        if s_arr.shape[0] == 1 and s_arr.shape[1] == X.shape[1]:
            # (1, n_features, n_classes)
            s_vals = s_arr[0, :, pred_label]
        elif s_arr.shape[1] == 1:
            # (n_classes, 1, n_features)
            s_vals = s_arr[pred_label, 0, :]
        else:
            s_vals = s_arr[0, :, pred_label]
    elif s_arr.ndim == 2:
        if s_arr.shape[0] == 1:
            # (1, n_features) - 이진 분류
            s_vals = s_arr[0]
        elif s_arr.shape[0] == X.shape[1]:
            # (n_features, n_classes)
            s_vals = s_arr[:, pred_label]
        else:
            s_vals = s_arr[0]
    else:
        s_vals = s_arr

    s_vals = np.array(s_vals).flatten()
    print(f"   ★ s_vals shape 변환 후: {s_vals.shape}")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    contrib = pd.Series(s_vals, index=X.columns).abs().sort_values(ascending=False).head(5)
    sign_vals = pd.Series(s_vals, index=X.columns)

    print("   [핵심 기여 변수 Top 5]")
    for feat in contrib.index:
        val = sign_vals[feat]
        direction = "↑ 불평등 증가" if val > 0 else "↓ 불평등 감소"
        print(f"     - {feat}: SHAP={val:+.4f} ({direction})")

# ============================================
# Phase 4: What-if 시뮬레이션 (수정본)
# ============================================

def simulate_policy(
    df: pd.DataFrame,
    X: pd.DataFrame,
    best_model,
    le: LabelEncoder,
    school_idx: int,
    budget_increase_pct: float = 0,
    afterschool_add: float = 0,
    facility_improve_pct: float = 0
) -> dict:
    """
    학교에 정책을 적용했을 때 불평등 등급 변화를 예측합니다.
    """
    school_data = X.loc[[school_idx]].copy()
    school_name = df.loc[school_idx, '학교명']

    # 원본 예측
    original_pred = le.inverse_transform(best_model.predict(school_data))[0]

    # 정책 적용
    if '학생1인당_수입' in school_data.columns and budget_increase_pct > 0:
        school_data['학생1인당_수입'] *= (1 + budget_increase_pct / 100)

    if '학생백명당_방과후프로그램' in school_data.columns and afterschool_add > 0:
        school_data['학생백명당_방과후프로그램'] += afterschool_add

    if facility_improve_pct > 0:
        for col in ['학생1인당_전체공간', '학생1인당_교실', '학생1인당_화장실']:
            if col in school_data.columns:
                school_data[col] *= (1 + facility_improve_pct / 100)

    # 변경 후 예측
    new_pred = le.inverse_transform(best_model.predict(school_data))[0]

    # ✅ 수정: 확률 변수를 안전하게 초기화
    result: dict = {
        '학교명': school_name,
        '정책_전_등급': original_pred,
        '정책_후_등급': new_pred,
        '등급_변화': original_pred != new_pred,
    }

    # predict_proba가 가능한 경우에만 확률 추가
    if hasattr(best_model, 'predict_proba'):
        orig_proba = best_model.predict_proba(X.loc[[school_idx]])[0]
        new_proba = best_model.predict_proba(school_data)[0]

        # ✅ 수정: .tolist()로 zip 호환 보장
        proba_labels: list[str] = le.classes_.tolist()
        result['정책_전_확률'] = {
            label: round(float(p), 3) for label, p in zip(proba_labels, orig_proba)
        }
        result['정책_후_확률'] = {
            label: round(float(p), 3) for label, p in zip(proba_labels, new_proba)
        }
    else:
        result['정책_전_확률'] = {}
        result['정책_후_확률'] = {}

    return result

# ── 시뮬레이션 실행 ──
print("=" * 10)
print("▶ What-if 정책 시뮬레이션")
print("=" * 10)

vulnerable_schools = df[df['교육불평등_등급'].isin(['취약', '심각'])].index.tolist()
if len(vulnerable_schools) == 0:
    threshold = df['교육불평등_복합지수'].quantile(0.9)
    vulnerable_schools = df[df['교육불평등_복합지수'] >= threshold].index.tolist()

scenarios = [
    {"name": "시나리오 1: 재정 20% 확대",
     "params": {"budget_increase_pct": 20}},
    {"name": "시나리오 2: 방과후 프로그램 5개 추가",
     "params": {"afterschool_add": 5}},
    {"name": "시나리오 3: 시설 30% 개선",
     "params": {"facility_improve_pct": 30}},
    {"name": "시나리오 4: 종합 패키지 (재정15% + 방과후3개 + 시설20%)",
     "params": {"budget_increase_pct": 15, "afterschool_add": 3, "facility_improve_pct": 20}},
]

results: list[dict] = []
for scenario in scenarios:
    print(f"\n📋 {scenario['name']}")
    print("-" * 10)

    improved = 0
    total = min(len(vulnerable_schools), 20)

    for idx in vulnerable_schools[:total]:
        if idx in X.index:
            result = simulate_policy(df, X, best_model, le, idx, **scenario['params'])
            if result['등급_변화']:
                improved += 1
                print(f"  ✅ {result['학교명']}: {result['정책_전_등급']} → {result['정책_후_등급']}")

    improvement_rate = improved / total * 100 if total > 0 else 0
    print(f"\n  📊 개선율: {improved}/{total} ({improvement_rate:.1f}%)")
    results.append({
        '시나리오': scenario['name'],
        '대상학교수': total,
        '개선학교수': improved,
        '개선율(%)': round(improvement_rate, 1)
    })

print("\n\n" + "=" * 10)
print("▶ 시나리오별 개선 효과 요약")
print("=" * 10)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ============================================
# Phase 5: 교육불평등 지수 상위 5개 학교 분석
# ============================================
print("\n" + "=" * 10)
print("▶ 교육불평등 지수 상위 5개 학교 분석")
print("=" * 10)

explainer_full = shap.TreeExplainer(best_model)
shap_full = np.array(explainer_full.shap_values(X))
print(f"▶ shap_full shape: {shap_full.shape}")
print(f"▶ X shape: {X.shape}")

top5 = df.nlargest(5, '교육불평등_복합지수')

for rank, (idx, row) in enumerate(top5.iterrows(), 1):
    school = row.get('학교명', f'학교_{idx}')
    region = str(row.get('시도', '')) + ' ' + str(row.get('시군구', ''))
    grade  = row.get('등급', 'N/A')
    score  = row.get('교육불평등_복합지수', 0)

    print(f"\n🏫 {school} ({region.strip()})")
    print(f"   등급: {grade}, 교육불평등_복합지수: {score:.4f}")

    if idx in X.index:
        x_pos = X.index.get_loc(idx)
    else:
        print(f"   ⚠️ 인덱스 매핑 실패, 건너뜁니다.")
        continue

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 핵심: 어떤 shape이든 강제로 1차원 (n_features,) 추출
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    raw = shap_full[x_pos]           # 이 시점에서 shape 확인
    print(f"   ▶ raw SHAP shape: {raw.shape}")

    if raw.ndim == 2:
        # (n_features, n_classes) → 예측 클래스의 SHAP만 추출
        pred_label = int(best_model.predict(X.iloc[[x_pos]])[0])
        s_vals = raw[:, pred_label]   # (n_features,)
        print(f"   ▶ 예측 클래스: {pred_label} → s_vals shape: {s_vals.shape}")
    elif raw.ndim == 1:
        s_vals = raw
        pred_label = int(best_model.predict(X.iloc[[x_pos]])[0])
    else:
        print(f"   ⚠️ 예상치 못한 shape: {raw.shape}, 건너뜁니다.")
        continue

    # s_vals가 확실히 1차원인지 최종 확인
    s_vals = np.array(s_vals).flatten()
    assert s_vals.shape == (X.shape[1],), f"shape 불일치: {s_vals.shape} != ({X.shape[1]},)"

    # 상위 5개 기여 변수 출력
    contrib = pd.Series(s_vals, index=X.columns).abs().sort_values(ascending=False).head(5)
    print(f"   📊 주요 기여 변수 (Top 5):")
    for feat_name, feat_val in contrib.items():
        raw_val = s_vals[X.columns.get_loc(feat_name)]
        direction = "▲ 취약↑" if raw_val > 0 else "▼ 양호↑"
        print(f"      - {feat_name}: |SHAP|={feat_val:.4f} ({direction})")

    # Waterfall Plot
    try:
        ev = explainer_full.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
            base_val = float(ev[pred_label])
        else:
            base_val = float(ev) if np.isscalar(ev) else float(ev[0])

        explanation = shap.Explanation(
            values=s_vals,
            base_values=base_val,
            data=X.iloc[x_pos].values,
            feature_names=X.columns.tolist()
        )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        pred_name = le.inverse_transform([pred_label])[0]
        plt.title(f'{school} SHAP Waterfall (예측: {pred_name})', fontsize=12)
        plt.tight_layout()
        fname = f'Waterfall/06_waterfall_{rank}_{school}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        #plt.show()
        print(f"   ✅ Waterfall 저장: {fname}")
    except Exception as e:
        print(f"   ⚠️ Waterfall 생성 실패: {e}")
        
# save_artifacts.py
# 기존 SHAP.py 파이프라인 실행 후, 맨 끝에 추가하여 1회 실행

import joblib
import os
import json
import numpy as np

os.makedirs('models', exist_ok=True)

# 1) 최적 모델 저장
joblib.dump(best_model, 'models/xgboost_model.pkl')

# 2) LabelEncoder 저장
joblib.dump(le, 'models/label_encoder.pkl')

# 3) 피처 목록 저장
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

# 4) 모델 성능 메트릭 저장
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

metrics_dict = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics_dict[name] = {
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'train_f1': float(f1_score(y_train, y_pred_train, average='weighted')),
        'test_f1': float(f1_score(y_test, y_pred, average='weighted')),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=le.classes_.tolist(),
            output_dict=True
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

joblib.dump(metrics_dict, 'models/metrics.pkl')

# 5) 학습/테스트 인덱스 저장
joblib.dump({
    'X_train_index': X_train.index.tolist(),
    'X_test_index': X_test.index.tolist(),
}, 'models/split_indices.pkl')

# 6) SHAP 값 저장 (전체 데이터 기준)
import shap
explainer = shap.TreeExplainer(best_model)
shap_values_all = explainer.shap_values(X)
joblib.dump({
    'shap_values': np.array(shap_values_all),
    'expected_value': explainer.expected_value,
    'feature_names': X.columns.tolist(),
    'X_values': X.values,
    'X_index': X.index.tolist(),
}, 'models/shap_data.pkl')

print("✅ 모든 아티팩트 저장 완료")
