import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import init_session

st.set_page_config(page_title="학교별 진단", page_icon="🏫", layout="wide")

ctx = init_session()
df = ctx['df']
NUMERIC_FEATURES = ctx['NUMERIC_FEATURES']

st.markdown("# 🏫 개별 학교 심층 진단")
st.markdown("학교를 선택하면 해당 학교의 상세 진단 결과를 확인할 수 있습니다.")
st.markdown("---")

# ============================================================
# 학교 선택
# ============================================================
col_s1, col_s2, col_s3 = st.columns([1, 1, 2])

with col_s1:
    if '시도교육청' in df.columns:
        region_list = ['전체'] + sorted(df['시도교육청'].dropna().unique().tolist())
        region_filter = st.selectbox("시도교육청", region_list)
    else:
        region_filter = '전체'

with col_s2:
    if '학교급' in df.columns:
        type_list = ['전체'] + df['학교급'].dropna().unique().tolist()
        type_filter = st.selectbox("학교급", type_list)
    else:
        type_filter = '전체'

with col_s3:
    subset = df.copy()
    if region_filter != '전체' and '시도교육청' in df.columns:
        subset = subset[subset['시도교육청'] == region_filter]
    if type_filter != '전체' and '학교급' in df.columns:
        subset = subset[subset['학교급'] == type_filter]

    school_col = '학교명' if '학교명' in df.columns else df.columns[0]
    school_list = sorted(subset[school_col].dropna().unique().tolist())
    selected_school = st.selectbox("학교 선택", school_list)

school_row = df[df[school_col] == selected_school].iloc[0]
school_idx = df[df[school_col] == selected_school].index[0]

# ============================================================
# 학교 정보 카드
# ============================================================
st.markdown("---")
info_cols = st.columns(5)
grade_icons = {'양호': '🟢', '보통': '🟡', '주의': '🟠', '취약': '🔴', '매우취약': '🔴', '심각': '⛔'}

with info_cols[0]:
    st.metric("학교명", school_row.get(school_col, 'N/A'))
with info_cols[1]:
    st.metric("시도교육청", school_row.get('시도교육청', 'N/A'))
with info_cols[2]:
    st.metric("학교급", school_row.get('학교급', 'N/A'))
with info_cols[3]:
    if '지역' in df.columns:
        st.metric("지역", school_row.get('지역', 'N/A'))
with info_cols[4]:
    grade = str(school_row.get('교육불평등_등급', 'N/A'))
    icon = grade_icons.get(grade, '⚪')
    st.metric("불평등 등급", f"{icon} {grade}")

if '교육불평등_복합지수' in df.columns:
    score = school_row['교육불평등_복합지수']
    percentile = (df['교육불평등_복합지수'] < score).mean() * 100
    st.markdown(f"**교육불평등 복합지수**: `{score:.4f}` (전국 상위 {100-percentile:.0f}%)")

# ============================================================
# 지표별 전국 대비 게이지 차트
# ============================================================
st.markdown("## 📊 전국 대비 지표 비교")
compare_vars = [c for c in NUMERIC_FEATURES if c in df.columns]

if compare_vars:
    n_cols = 3
    for row_start in range(0, len(compare_vars), n_cols):
        gauge_cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = row_start + j
            if idx >= len(compare_vars):
                break
            var = compare_vars[idx]
            with gauge_cols[j]:
                val = float(school_row[var]) if pd.notna(school_row[var]) else 0
                min_val = float(df[var].min())
                max_val = float(df[var].max())
                avg_val = float(df[var].mean())
                pct = float((df[var] < val).mean() * 100)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=val,
                    delta={'reference': avg_val, 'relative': True, 'valueformat': '.1%'},
                    title={'text': var, 'font': {'size': 12}},
                    gauge={
                        'axis': {'range': [min_val, max_val]},
                        'bar': {'color': '#667eea'},
                        'steps': [
                            {'range': [min_val, avg_val], 'color': '#FFF3E0'},
                            {'range': [avg_val, max_val], 'color': '#E8F5E9'}
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 3},
                            'thickness': 0.8, 'value': avg_val
                        }
                    }
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption(f"전국 백분위: **{pct:.0f}%** (평균: {avg_val:.2f})")

# ============================================================
# 같은 지역 유사 학교와 비교
# ============================================================
st.markdown("## 🔄 동일 지역 내 위치 비교")

if '시도교육청' in df.columns and '학교급' in df.columns:
    same_region = df[
        (df['시도교육청'] == school_row.get('시도교육청', '')) &
        (df['학교급'] == school_row.get('학교급', ''))
    ].copy()

    compare_var = st.selectbox("비교 변수 선택", compare_vars, key='school_compare_var')

    fig_compare = px.histogram(
        same_region, x=compare_var, nbins=20,
        color_discrete_sequence=['#B0BEC5'], opacity=0.6,
        title=f"{school_row.get('시도교육청', '')} {school_row.get('학교급', '')} ({len(same_region)}개교)"
    )
    school_val = float(school_row[compare_var]) if pd.notna(school_row[compare_var]) else 0
    region_avg = float(same_region[compare_var].mean())

    fig_compare.add_vline(
        x=school_val, line_dash="dash", line_color="red", line_width=3,
        annotation_text=f"▸ {selected_school}: {school_val:.2f}", annotation_position="top"
    )
    fig_compare.add_vline(
        x=region_avg, line_dash="dot", line_color="blue", line_width=2,
        annotation_text=f"▸ 지역 평균: {region_avg:.2f}", annotation_position="bottom"
    )
    fig_compare.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_compare, use_container_width=True)

with st.expander("📋 해당 학교 전체 데이터 보기"):
    school_detail = pd.DataFrame(school_row).T
    st.dataframe(school_detail, use_container_width=True)