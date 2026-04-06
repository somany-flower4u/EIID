import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import init_session

st.set_page_config(page_title="데이터 탐색", page_icon="📊", layout="wide")

# ✅ 어느 페이지에서 진입하든 데이터/모델이 준비됨
ctx = init_session()
df = ctx['df']
NUMERIC_FEATURES = ctx['NUMERIC_FEATURES']

st.markdown("# 📊 데이터 탐색 (EDA)")
st.markdown("전체 데이터의 분포와 패턴을 시각적으로 탐색합니다.")
st.markdown("---")

# ============================================================
# 1. 기술통계
# ============================================================
st.markdown("## 1️⃣ 기술통계 요약")

target_and_features = [c for c in ['교육불평등_복합지수'] + NUMERIC_FEATURES if c in df.columns]
if target_and_features:
    desc = df[target_and_features].describe().round(3).T
    st.dataframe(desc.style.background_gradient(cmap='YlOrRd', axis=1), use_container_width=True)

st.markdown("### 🔍 결측치 현황")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'결측수': missing, '비율(%)': missing_pct})
missing_df = missing_df[missing_df['결측수'] > 0].sort_values('결측수', ascending=False)
if len(missing_df) > 0:
    st.dataframe(missing_df, use_container_width=True)
else:
    st.success("결측치가 없습니다.")

# ============================================================
# 2. 변수별 분포
# ============================================================
st.markdown("## 2️⃣ 변수별 분포")

dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    selected_var = st.selectbox("분석할 변수 선택", target_and_features, key='dist_var')
with dist_col2:
    bin_count = st.slider("히스토그램 구간 수", 10, 100, 30)

fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["히스토그램", "불평등 등급별 Box Plot"])

fig_dist.add_trace(
    go.Histogram(x=df[selected_var].dropna(), nbinsx=bin_count,
                 name="분포", marker_color='#667eea', opacity=0.7),
    row=1, col=1
)

if '교육불평등_등급' in df.columns:
    for grade in df['교육불평등_등급'].dropna().unique():
        subset = df[df['교육불평등_등급'] == grade]
        fig_dist.add_trace(go.Box(y=subset[selected_var].dropna(), name=str(grade)), row=1, col=2)

fig_dist.update_layout(height=450, showlegend=False, margin=dict(t=40, b=0))
st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# 3. 상관관계 히트맵
# ============================================================
st.markdown("## 3️⃣ 상관관계 히트맵")

corr_vars = st.multiselect(
    "변수 선택", target_and_features,
    default=target_and_features[:min(8, len(target_and_features))]
)

if len(corr_vars) >= 2:
    corr_matrix = df[corr_vars].corr()
    fig_heatmap = px.imshow(
        corr_matrix.round(3), text_auto=True,
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto'
    )
    fig_heatmap.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    if '교육불평등_복합지수' in corr_vars:
        st.markdown("#### 📊 교육불평등_복합지수와의 상관관계 (절대값 순)")
        corr_target = corr_matrix['교육불평등_복합지수'].drop('교육불평등_복합지수', errors='ignore')
        corr_sorted = corr_target.abs().sort_values(ascending=False)
        corr_display = pd.DataFrame({
            '변수': corr_sorted.index,
            '상관계수': [corr_target[v] for v in corr_sorted.index],
            '|상관계수|': corr_sorted.values
        })
        st.dataframe(corr_display, use_container_width=True)
else:
    st.info("2개 이상의 변수를 선택해주세요.")

# ============================================================
# 4. 지역별 비교 레이더 차트
# ============================================================
st.markdown("## 4️⃣ 시도교육청별 비교 (레이더 차트)")

if '시도교육청' in df.columns:
    radar_regions = st.multiselect(
        "비교할 시도교육청 선택 (최대 5개)",
        sorted(df['시도교육청'].dropna().unique().tolist()),
        default=sorted(df['시도교육청'].dropna().unique().tolist())[:3],
        max_selections=5
    )
    radar_vars = [c for c in NUMERIC_FEATURES if c in df.columns][:6]

    if radar_regions and len(radar_vars) >= 3:
        fig_radar = go.Figure()
        for region in radar_regions:
            region_data = df[df['시도교육청'] == region][radar_vars].mean()
            col_min = df[radar_vars].min()
            col_max = df[radar_vars].max()
            col_range = (col_max - col_min).replace(0, 1)
            normalized = (region_data - col_min) / col_range
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized.values.tolist() + [normalized.values[0]],
                theta=radar_vars + [radar_vars[0]],
                name=region, fill='toself', opacity=0.5
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500, margin=dict(l=80, r=80, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)