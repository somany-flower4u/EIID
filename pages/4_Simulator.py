import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import init_session

st.set_page_config(page_title="정책 시뮬레이션", page_icon="🧪", layout="wide")

# ✅ 어느 페이지에서 바로 진입해도 데이터/모델 로드됨
ctx = init_session()
df = ctx['df']
model = ctx['model']
le = ctx['le']
X = ctx['X']
NUMERIC_FEATURES = ctx['NUMERIC_FEATURES']

st.markdown("# 🧪 What-if 정책 시뮬레이션")
st.markdown("정책 시나리오를 설정하면 모델이 예측하는 불평등 등급 변화를 시뮬레이션합니다.")
st.markdown("---")

# ============================================================
# 시나리오 설정
# ============================================================
st.markdown("## ⚙️ 시나리오 설정")

scenario_cols = st.columns(3)

with scenario_cols[0]:
    st.markdown("### 💰 재정 확대")
    income_increase = st.slider(
        "학생1인당 수입 증가율 (%)", min_value=0, max_value=100, value=20, step=5
    )

with scenario_cols[1]:
    st.markdown("### 📚 방과후 프로그램 확충")
    program_add = st.slider(
        "학생백명당 방과후프로그램 추가 수", min_value=0.0, max_value=30.0, value=5.0, step=0.5
    )

with scenario_cols[2]:
    st.markdown("### 🏗️ 시설 개선")
    facility_increase = st.slider(
        "시설 면적 증가율 (%)", min_value=0, max_value=100, value=30, step=5
    )

st.markdown("---")

# 적용 대상
target_option = st.radio(
    "시나리오 적용 대상",
    ["전체 학교", "취약 등급 이상 학교만", "특정 시도교육청만"],
    horizontal=True
)

specific_region = None
if target_option == "특정 시도교육청만" and '시도교육청' in df.columns:
    specific_region = st.selectbox(
        "시도교육청 선택", sorted(df['시도교육청'].dropna().unique().tolist())
    )

# ============================================================
# 시뮬레이션 실행
# ============================================================
if st.button("🚀 시뮬레이션 실행", type="primary", use_container_width=True):

    valid_idx = X.index.intersection(df.index)

    if target_option == "취약 등급 이상 학교만":
        vuln_labels = ['취약', '매우취약', '심각']
        target_schools = df.loc[valid_idx]
        target_schools = target_schools[target_schools['교육불평등_등급'].isin(vuln_labels)]
        target_idx = target_schools.index
    elif target_option == "특정 시도교육청만" and specific_region:
        target_schools = df.loc[valid_idx]
        target_schools = target_schools[target_schools['시도교육청'] == specific_region]
        target_idx = target_schools.index
    else:
        target_idx = valid_idx

    if len(target_idx) == 0:
        st.warning("적용 대상 학교가 없습니다. 필터를 조정해주세요.")
        st.stop()

    st.markdown(f"**대상 학교: {len(target_idx):,}개**")

    # 원본 예측
    X_original = X.loc[target_idx].copy()
    original_preds = le.inverse_transform(model.predict(X_original))

    # 시나리오 적용
    X_sim = X_original.copy()

    if income_increase > 0 and '학생1인당_수입' in X_sim.columns:
        X_sim['학생1인당_수입'] *= (1 + income_increase / 100)

    if program_add > 0 and '학생백명당_방과후프로그램' in X_sim.columns:
        X_sim['학생백명당_방과후프로그램'] += program_add

    if facility_increase > 0:
        for col in ['학생1인당_전체공간', '학생1인당_교실', '학생1인당_화장실']:
            if col in X_sim.columns:
                X_sim[col] *= (1 + facility_increase / 100)

    # 시뮬레이션 예측
    sim_preds = le.inverse_transform(model.predict(X_sim))

    # 결과 DataFrame
    info_cols_list = [c for c in ['학교명', '시도교육청', '지역', '학교급', '교육불평등_등급', '교육불평등_복합지수']
                      if c in df.columns]
    result_df = df.loc[target_idx, info_cols_list].copy()
    result_df['원래_예측'] = original_preds
    result_df['시뮬_예측'] = sim_preds
    result_df['변화여부'] = result_df['원래_예측'] != result_df['시뮬_예측']

    # 확률 변화
    if hasattr(model, 'predict_proba'):
        orig_proba = model.predict_proba(X_original)
        sim_proba = model.predict_proba(X_sim)

        classes_list = le.classes_.tolist()
        vuln_candidates = ['취약', '매우취약', '심각']
        vuln_idx_list = [classes_list.index(c) for c in vuln_candidates if c in classes_list]

        if vuln_idx_list:
            orig_vuln_prob = orig_proba[:, vuln_idx_list].sum(axis=1)
            sim_vuln_prob = sim_proba[:, vuln_idx_list].sum(axis=1)
            result_df['취약확률_전'] = (orig_vuln_prob * 100).round(1)
            result_df['취약확률_후'] = (sim_vuln_prob * 100).round(1)
            result_df['확률변화'] = (result_df['취약확률_후'] - result_df['취약확률_전']).round(1)

    # ---- 결과 시각화 ----
    st.markdown("## 📊 시뮬레이션 결과")

    res_cols = st.columns(2)

    with res_cols[0]:
        st.markdown("### 🔴 현재 (Before)")
        before_counts = pd.Series(original_preds).value_counts()
        fig_before = px.pie(values=before_counts.values, names=before_counts.index, hole=0.4)
        fig_before.update_traces(textinfo='label+value')
        fig_before.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_before, use_container_width=True)

    with res_cols[1]:
        st.markdown("### 🟢 시뮬레이션 (After)")
        after_counts = pd.Series(sim_preds).value_counts()
        fig_after = px.pie(values=after_counts.values, names=after_counts.index, hole=0.4)
        fig_after.update_traces(textinfo='label+value')
        fig_after.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_after, use_container_width=True)

    # 개선 통계
    st.markdown("### 📈 개선 요약")

    improved_count = int(result_df['변화여부'].sum())
    total_count = len(result_df)
    improvement_rate = improved_count / total_count * 100 if total_count > 0 else 0

    vuln_labels = ['취약', '매우취약', '심각']
    before_vuln = int((pd.Series(original_preds).isin(vuln_labels)).sum())
    after_vuln = int((pd.Series(sim_preds).isin(vuln_labels)).sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("대상 학교", f"{total_count:,}개")
    metric_cols[1].metric("등급 변화 학교", f"{improved_count:,}개", delta=f"{improvement_rate:.1f}%")
    metric_cols[2].metric("취약 이상 (Before)", f"{before_vuln:,}개")
    metric_cols[3].metric("취약 이상 (After)", f"{after_vuln:,}개", delta=f"{after_vuln - before_vuln}개")

    # 등급 변동 상세 (Sankey Diagram)
    st.markdown("### 🔀 등급 변동 흐름")

    before_series = pd.Series(original_preds)
    after_series = pd.Series(sim_preds)
    flow_df = pd.DataFrame({'Before': before_series, 'After': after_series})
    flow_counts = flow_df.groupby(['Before', 'After']).size().reset_index(name='학교수')

    all_labels_unique = list(set(flow_counts['Before'].tolist() + flow_counts['After'].tolist()))
    before_labels = [f"Before: {l}" for l in all_labels_unique]
    after_labels = [f"After: {l}" for l in all_labels_unique]
    all_sankey_labels = before_labels + after_labels

    source_indices = [before_labels.index(f"Before: {b}") for b in flow_counts['Before']]
    target_indices = [len(before_labels) + after_labels.index(f"After: {a}") for a in flow_counts['After']]

    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_sankey_labels,
            color=['#EF5350' if '취약' in l or '심각' in l else '#42A5F5' if '양호' in l else '#FFA726'
                   for l in all_sankey_labels]
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=flow_counts['학교수'].tolist(),
            color='rgba(100,126,234,0.3)'
        )
    ))
    fig_sankey.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_sankey, use_container_width=True)

    # 변화된 학교 목록
    st.markdown("### 📋 등급이 변화한 학교")
    changed_df = result_df[result_df['변화여부']].copy()

    if len(changed_df) > 0:
        sort_col = '확률변화' if '확률변화' in changed_df.columns else '원래_예측'
        st.dataframe(changed_df.sort_values(sort_col), use_container_width=True, height=400)
    else:
        st.info("현재 시나리오에서는 등급이 변화한 학교가 없습니다. 정책 강도를 높여보세요.")

    # 확률 변화 분포
    if '확률변화' in result_df.columns:
        st.markdown("### 📊 취약 확률 변화 분포")
        fig_prob = px.histogram(
            result_df, x='확률변화', nbins=30,
            color_discrete_sequence=['#667eea'], opacity=0.7,
            labels={'확률변화': '취약 확률 변화 (%p)'}
        )
        fig_prob.add_vline(x=0, line_dash="dash", line_color="red")
        fig_prob.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_prob, use_container_width=True)

    # CSV 다운로드
    csv = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 시뮬레이션 결과 다운로드",
        data=csv, file_name='정책_시뮬레이션_결과.csv', mime='text/csv'
    )