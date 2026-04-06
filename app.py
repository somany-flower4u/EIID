import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import init_session

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="교육 불평등 분석 대시보드",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 커스텀 CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        color: #1B3A5C; text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem; color: #666;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1.5rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.5rem 0 0 0; font-size: 2.2rem; }
    .metric-card-blue   { background: linear-gradient(135deg, #2196F3, #21CBF3); }
    .metric-card-green  { background: linear-gradient(135deg, #43A047, #66BB6A); }
    .metric-card-orange { background: linear-gradient(135deg, #FF6B35, #F7C948); }
    .metric-card-red    { background: linear-gradient(135deg, #E53935, #EF5350); }
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border: none; border-radius: 2px; margin: 1rem 0;
    }
    [data-testid="stSidebar"] { background-color: #F8F9FA; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 공통 초기화 (데이터, 모델, 피처 매트릭스 한 번에 로드)
# ============================================================
ctx = init_session()
df = ctx['df']
NUMERIC_FEATURES = ctx['NUMERIC_FEATURES']

# ============================================================
# 사이드바 필터
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 교육 불평등 분석")
    st.markdown("---")
    st.markdown("### 필터 설정")

    if '시도교육청' in df.columns:
        all_regions = sorted(df['시도교육청'].dropna().unique().tolist())
        selected_regions = st.multiselect(
            "시도교육청 선택", all_regions, default=all_regions, key='region_filter'
        )
    else:
        selected_regions = None

    if '학교급' in df.columns:
        all_types = df['학교급'].dropna().unique().tolist()
        selected_types = st.multiselect(
            "학교급 선택", all_types, default=all_types, key='type_filter'
        )
    else:
        selected_types = None

    if '교육불평등_등급' in df.columns:
        all_grades = df['교육불평등_등급'].dropna().unique().tolist()
        selected_grades = st.multiselect(
            "불평등 등급 선택", all_grades, default=all_grades, key='grade_filter'
        )
    else:
        selected_grades = None

    st.markdown("---")
    st.caption(f"📋 전체 데이터: {len(df):,}개 학교, {df.shape[1]}개 변수")

# 필터 적용
mask = pd.Series([True] * len(df), index=df.index)
if selected_regions is not None and '시도교육청' in df.columns:
    mask &= df['시도교육청'].isin(selected_regions)
if selected_types is not None and '학교급' in df.columns:
    mask &= df['학교급'].isin(selected_types)
if selected_grades is not None and '교육불평등_등급' in df.columns:
    mask &= df['교육불평등_등급'].isin(selected_grades)

filtered_df = df[mask].copy()
st.session_state['filtered_df'] = filtered_df

# ============================================================
# 메인 콘텐츠
# ============================================================
st.markdown('<p class="main-header">🎓 교육 불평등 분석 대시보드</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">전국 학교 데이터를 기반으로 교육 불평등을 진단하고 정책 시뮬레이션을 수행합니다</p>',
    unsafe_allow_html=True
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---- KPI 카드 ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card metric-card-blue">
        <h3>분석 대상 학교</h3>
        <h1>{len(filtered_df):,}</h1>
    </div>""", unsafe_allow_html=True)

with col2:
    vuln_labels = ['취약', '매우취약', '심각']
    if '교육불평등_등급' in filtered_df.columns:
        vuln_count = len(filtered_df[filtered_df['교육불평등_등급'].isin(vuln_labels)])
    else:
        vuln_count = 0
    st.markdown(f"""
    <div class="metric-card metric-card-red">
        <h3>취약 이상 학교</h3>
        <h1>{vuln_count:,}</h1>
    </div>""", unsafe_allow_html=True)

with col3:
    if '학생1인당_수입' in filtered_df.columns:
        avg_income = filtered_df['학생1인당_수입'].mean()
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <h3>평균 학생1인당 수입</h3>
            <h1>{avg_income:.1f}</h1>
        </div>""", unsafe_allow_html=True)

with col4:
    if '학생백명당_방과후프로그램' in filtered_df.columns:
        avg_prog = filtered_df['학생백명당_방과후프로그램'].mean()
        st.markdown(f"""
        <div class="metric-card metric-card-orange">
            <h3>100명당 방과후 프로그램</h3>
            <h1>{avg_prog:.1f}</h1>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- 1행: 지역별 불평등 분포 + 등급 파이차트 ----
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.markdown("### 📍 시도교육청별 평균 불평등 지수")
    if '시도교육청' in filtered_df.columns and '교육불평등_복합지수' in filtered_df.columns:
        region_stats = filtered_df.groupby('시도교육청').agg(
            평균_불평등지수=('교육불평등_복합지수', 'mean'),
            학교수=('교육불평등_복합지수', 'count')
        ).reset_index().sort_values('평균_불평등지수', ascending=True)

        fig_bar = px.bar(
            region_stats, x='평균_불평등지수', y='시도교육청',
            orientation='h', color='평균_불평등지수',
            color_continuous_scale='RdYlGn_r',
            text=region_stats['평균_불평등지수'].round(3),
            hover_data=['학교수']
        )
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(
            height=500, margin=dict(l=0, r=40, t=10, b=0),
            coloraxis_showscale=False, xaxis_title="평균 복합지수", yaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with row1_col2:
    st.markdown("### 📊 불평등 등급 분포")
    if '교육불평등_등급' in filtered_df.columns:
        grade_counts = filtered_df['교육불평등_등급'].value_counts().reset_index()
        grade_counts.columns = ['등급', '학교수']
        fig_pie = px.pie(
            grade_counts, values='학교수', names='등급',
            color_discrete_sequence=px.colors.qualitative.Set1, hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

# ---- 2행: 산점도 + 학교급별 비교 ----
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("### 🔗 주요 변수 간 산점도")
    available_numeric = [c for c in NUMERIC_FEATURES + ['교육불평등_복합지수'] if c in filtered_df.columns]
    x_var = st.selectbox("X축 변수", available_numeric,
                         index=available_numeric.index('학생1인당_수입') if '학생1인당_수입' in available_numeric else 0,
                         key='scatter_x')
    y_var = st.selectbox("Y축 변수", available_numeric,
                         index=available_numeric.index('교육불평등_복합지수') if '교육불평등_복합지수' in available_numeric else 0,
                         key='scatter_y')
    fig_scatter = px.scatter(
        filtered_df, x=x_var, y=y_var,
        color='교육불평등_등급' if '교육불평등_등급' in filtered_df.columns else None,
        opacity=0.6,
        hover_data=['학교명'] if '학교명' in filtered_df.columns else None
    )
    fig_scatter.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_col2:
    st.markdown("### 🏫 학교급별 불평등 비교")
    if '학교급' in filtered_df.columns and '교육불평등_복합지수' in filtered_df.columns:
        fig_box = px.box(
            filtered_df, x='학교급', y='교육불평등_복합지수',
            color='학교급', color_discrete_sequence=px.colors.qualitative.Set2, points='outliers'
        )
        fig_box.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, yaxis_title="교육불평등 복합지수", xaxis_title=""
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ---- 3행: 데이터 테이블 ----
st.markdown("### 📋 학교 데이터 테이블")
default_show = [c for c in ['학교명', '시도교육청', '지역', '학교급', '학생1인당_수입',
                             '학생백명당_방과후프로그램', '교육불평등_복합지수', '교육불평등_등급']
                if c in filtered_df.columns]
show_cols = st.multiselect("표시할 컬럼 선택", options=filtered_df.columns.tolist(), default=default_show)

if show_cols:
    sort_col = '교육불평등_복합지수' if '교육불평등_복합지수' in show_cols else show_cols[0]
    st.dataframe(
        filtered_df[show_cols].sort_values(sort_col, ascending=False),
        use_container_width=True, height=400
    )

csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    label="📥 필터링 데이터 CSV 다운로드",
    data=csv, file_name='교육불평등_필터링데이터.csv', mime='text/csv'
)

# ============================================================
# 페이지 네비게이션 카드 (수정된 부분)
# ============================================================
st.markdown("---")
st.markdown("## 🧭 분석 도구 바로가기")

nav_cols = st.columns(5)

nav_items = [
    {
        "page": "pages/1_Data_Search.py",
        "icon": "📊", "title": "데이터 탐색",
        "desc": "전체 데이터의 분포와 패턴을 시각적으로 탐색합니다."
    },
    {
        "page": "pages/2_School_Diagnosis.py",
        "icon": "🏫", "title": "학교별 진단",
        "desc": "개별 학교의 상세 진단 결과를 확인합니다."
    },
    {
        "page": "pages/3_SHAP.py",
        "icon": "🔍", "title": "SHAP 분석",
        "desc": "모델이 어떤 변수를 중요하게 보는지 해석합니다."
    },
    {
        "page": "pages/4_Simulator.py",
        "icon": "🧪", "title": "정책 시뮬레이션",
        "desc": "What-if 시나리오로 정책 효과를 예측합니다."
    },
    {
        "page": "pages/5_Model_Performance.py",
        "icon": "📈", "title": "모델 성능",
        "desc": "분류 모델의 성능 평가 결과를 확인합니다."
    },
]

for col, item in zip(nav_cols, nav_items):
    with col:
        st.markdown(f"""
        <div style="
            background: white; border-radius: 12px; padding: 1.2rem;
            text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #eee; min-height: 160px;
        ">
            <div style="font-size: 2.5rem;">{item['icon']}</div>
            <div style="font-weight: 700; margin: 0.5rem 0; font-size: 1rem;">{item['title']}</div>
            <div style="color: #888; font-size: 0.8rem;">{item['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        # ✅ st.page_link 대신 st.button + st.switch_page 사용
        if st.button(f"{item['icon']} {item['title']} 열기", key=f"nav_{item['title']}", use_container_width=True):
            st.switch_page(item['page'])