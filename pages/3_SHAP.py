import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
from utils.data_loader import init_session

st.set_page_config(page_title="SHAP 분석", page_icon="🔍", layout="wide")

ctx = init_session()
df = ctx['df']
model = ctx['model']
le = ctx['le']
X = ctx['X']
shap_data = ctx['shap_data']

st.markdown("# 🔍 SHAP 기반 모델 해석")
st.markdown("모델이 불평등 등급을 예측할 때 어떤 변수가 얼마나 영향을 미치는지 해석합니다.")
st.markdown("---")

# ============================================================
# SHAP 값 계산 또는 로드
# ============================================================
@st.cache_data
def compute_shap_values(_model, _X_values, _X_columns):
    explainer = shap.TreeExplainer(_model)
    X_df = pd.DataFrame(_X_values, columns=_X_columns)
    shap_vals = explainer.shap_values(X_df)
    return np.array(shap_vals), explainer.expected_value

feature_names = X.columns.tolist()

if shap_data is not None:
    shap_arr = np.array(shap_data['shap_values'])
    expected_value = shap_data['expected_value']
    feature_names = shap_data['feature_names']
    st.success("✅ 저장된 SHAP 데이터 로드 완료")
else:
    with st.spinner("🔄 SHAP 값 계산 중... (최초 1회만 소요됩니다)"):
        shap_arr, expected_value = compute_shap_values(model, X.values, X.columns.tolist())
    st.success("✅ SHAP 값 계산 완료")

st.caption(f"SHAP 배열 shape: `{shap_arr.shape}`, 피처 수: `{len(feature_names)}`")

# ============================================================
# SHAP 차원 정리
# ============================================================
n_classes = len(le.classes_)

if shap_arr.ndim == 3:
    if shap_arr.shape[0] == len(X) and shap_arr.shape[1] == len(feature_names):
        shap_mean_global = np.abs(shap_arr).mean(axis=2).mean(axis=0)
    else:
        shap_mean_global = np.abs(shap_arr).mean(axis=1).mean(axis=0)
elif shap_arr.ndim == 2:
    shap_mean_global = np.abs(shap_arr).mean(axis=0)
else:
    shap_mean_global = np.abs(shap_arr).flatten()

# 타겟 클래스 결정
target_class_name = None
for candidate in ['취약', '매우취약', '심각']:
    if candidate in le.classes_:
        target_class_name = candidate
        break
if target_class_name is None:
    target_class_name = le.classes_[-1]

target_class_idx = list(le.classes_).index(target_class_name)

if shap_arr.ndim == 3:
    if shap_arr.shape[0] == len(X):
        shap_for_target = shap_arr[:, :, target_class_idx]
    else:
        shap_for_target = shap_arr[target_class_idx]
else:
    shap_for_target = shap_arr

# ============================================================
# 1. 전역 변수 중요도
# ============================================================
st.markdown("## 1️⃣ 전역 변수 중요도 (평균 |SHAP|)")

imp_df = pd.DataFrame({
    '변수': feature_names,
    '평균_SHAP': shap_mean_global
}).sort_values('평균_SHAP', ascending=True)

fig_imp = px.bar(
    imp_df.tail(15), x='평균_SHAP', y='변수',
    orientation='h', color='평균_SHAP',
    color_continuous_scale='Reds',
    labels={'평균_SHAP': '평균 |SHAP 값|', '변수': ''}
)
fig_imp.update_layout(height=500, margin=dict(l=0, r=20, t=10, b=0), coloraxis_showscale=False)
st.plotly_chart(fig_imp, use_container_width=True)

st.info(
    f"💡 **해석 가이드**: SHAP 값이 클수록 불평등 등급 예측에 더 큰 영향. "
    f"양(+)의 SHAP는 **'{target_class_name}'** 방향, 음(-)은 양호 방향."
)

# ============================================================
# 2. SHAP Beeswarm Plot (go.Scatter 기반)
# ============================================================
st.markdown(f"## 2️⃣ SHAP Beeswarm Plot (대상: '{target_class_name}' 클래스)")

top_n = st.slider("표시할 상위 변수 수", 5, min(20, len(feature_names)), 10)

top_features_idx = np.argsort(np.abs(shap_for_target).mean(axis=0))[::-1][:top_n]
top_feature_names = [feature_names[i] for i in top_features_idx]

max_samples = min(500, shap_for_target.shape[0])
np.random.seed(42)
sample_idx = np.random.choice(shap_for_target.shape[0], max_samples, replace=False)

beeswarm_rows = []
for feat_rank, feat_i in enumerate(top_features_idx):
    feat_name = feature_names[feat_i]
    if hasattr(X, 'iloc'):
        raw_vals = X.iloc[sample_idx, feat_i].values
    else:
        raw_vals = X[sample_idx, feat_i]
    raw_min, raw_max = np.nanmin(raw_vals), np.nanmax(raw_vals)
    norm_vals = (raw_vals - raw_min) / (raw_max - raw_min + 1e-10)

    for si in range(max_samples):
        actual_i = sample_idx[si]
        beeswarm_rows.append({
            '변수순위': feat_rank,
            'SHAP': float(shap_for_target[actual_i, feat_i]),
            'Feature_norm': float(norm_vals[si]),
            'Feature_raw': float(raw_vals[si]),
            '변수': feat_name
        })

beeswarm_df = pd.DataFrame(beeswarm_rows)
beeswarm_df['y_jitter'] = beeswarm_df['변수순위'] + np.random.uniform(-0.3, 0.3, size=len(beeswarm_df))

fig_bee = go.Figure()
fig_bee.add_trace(go.Scatter(
    x=beeswarm_df['SHAP'],
    y=beeswarm_df['y_jitter'],
    mode='markers',
    marker=dict(
        size=3, color=beeswarm_df['Feature_norm'],
        colorscale='RdBu_r', opacity=0.5, showscale=True,
        colorbar=dict(title=dict(text="Feature 값<br>(정규화)"), thickness=15, len=0.8),
        cmin=0, cmax=1
    ),
    customdata=np.stack([
        beeswarm_df['변수'], beeswarm_df['Feature_raw'].round(3), beeswarm_df['SHAP'].round(4)
    ], axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>SHAP: %{customdata[2]}<br>Feature: %{customdata[1]}<extra></extra>"
))
fig_bee.update_layout(
    yaxis=dict(tickmode='array', tickvals=list(range(top_n)), ticktext=top_feature_names, autorange='reversed'),
    xaxis_title="SHAP 값",
    height= 5 * top_n + 120,
    margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor='white'
)
fig_bee.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1)
st.plotly_chart(fig_bee, use_container_width=True)

# ============================================================
# 3. 개별 학교 SHAP Waterfall
# ============================================================
st.markdown("## 3️⃣ 개별 학교 SHAP 분석")

school_col = '학교명' if '학교명' in df.columns else df.columns[0]
available_schools = df[df.index.isin(X.index)][school_col].dropna().unique().tolist()
selected_school = st.selectbox("분석할 학교 선택", sorted(available_schools), key='shap_school')
school_idx = df[df[school_col] == selected_school].index[0]

if school_idx in X.index:
    x_pos = X.index.get_loc(school_idx)
    school_X = X.iloc[x_pos]

    if shap_arr.ndim == 3:
        pred_label = int(model.predict(X.iloc[[x_pos]])[0])
        if shap_arr.shape[0] == len(X):
            s_vals = shap_arr[x_pos, :, pred_label]
        else:
            s_vals = shap_arr[pred_label, x_pos, :]
    elif shap_arr.ndim == 2:
        s_vals = shap_arr[x_pos]
        pred_label = int(model.predict(X.iloc[[x_pos]])[0])
    else:
        s_vals = shap_arr
        pred_label = 0

    s_vals = np.array(s_vals).flatten()
    pred_name = le.inverse_transform([pred_label])[0]

    school_row = df.loc[school_idx]
    info_cols = st.columns(4)
    info_cols[0].metric("학교명", selected_school)
    info_cols[1].metric("실제 등급", str(school_row.get('교육불평등_등급', 'N/A')))
    info_cols[2].metric("예측 등급", str(pred_name))
    info_cols[3].metric("복합지수", f"{school_row.get('교육불평등_복합지수', 0):.4f}")

    contrib = pd.Series(s_vals, index=feature_names)
    top_contrib = contrib.abs().sort_values(ascending=False).head(10)
    top_names = top_contrib.index.tolist()
    top_values = [contrib[n] for n in top_names]

    fig_waterfall = go.Figure(go.Waterfall(
        name="SHAP", orientation="h",
        y=top_names[::-1], x=top_values[::-1],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#EF5350"}},
        decreasing={"marker": {"color": "#42A5F5"}},
        textposition="outside",
        text=[f"{v:+.4f}" for v in top_values[::-1]]
    ))
    fig_waterfall.update_layout(
        title=f"'{selected_school}' SHAP 기여도 Top 10 (예측: {pred_name})",
        height=400, margin=dict(l=0, r=100, t=40, b=0),
        xaxis_title="SHAP 값 (+ = 취약 방향, - = 양호 방향)"
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("#### 📝 자동 해석")
    for rank, feat_name in enumerate(top_names[:5], 1):
        val = contrib[feat_name]
        direction = "취약" if val > 0 else "양호"
        feat_val = school_X[feat_name] if feat_name in school_X.index else 'N/A'
        st.markdown(
            f"- **{rank}위 `{feat_name}`**: 실제값 = `{feat_val}`, "
            f"SHAP = `{val:+.4f}` → 등급을 **{direction}** 방향으로 기여"
        )
else:
    st.warning("해당 학교의 피처 데이터를 찾을 수 없습니다.")