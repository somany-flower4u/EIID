import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import init_session

st.set_page_config(page_title="모델 성능", page_icon="📈", layout="wide")

ctx = init_session()
le = ctx['le']
metrics_data = ctx['metrics']

st.markdown("# 📈 모델 성능 리포트")
st.markdown("학습된 분류 모델의 성능 평가 결과를 확인합니다.")
st.markdown("---")

if metrics_data is None:
    st.warning(
        "⚠️ `models/metrics.pkl` 파일이 없습니다. "
        "`save_artifacts.py`를 실행하여 모델 성능 데이터를 저장해주세요."
    )
    st.stop()

# ============================================================
# 1. 모델별 성능 비교
# ============================================================
st.markdown("## 1️⃣ 모델 성능 비교")

model_names = list(metrics_data.keys())
labels = le.classes_.tolist()

perf_rows = []
for name, m in metrics_data.items():
    perf_rows.append({
        'Model': name,
        'Train Accuracy': m.get('train_accuracy', 0),
        'Test Accuracy': m.get('test_accuracy', 0),
        'Train F1': m.get('train_f1', 0),
        'Test F1': m.get('test_f1', 0),
    })
perf_df = pd.DataFrame(perf_rows)

tab1, tab2 = st.tabs(["📊 차트", "📋 테이블"])

with tab1:
    metrics_list = ['Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1']
    fig_perf = make_subplots(rows=1, cols=4, subplot_titles=metrics_list)
    colors = px.colors.qualitative.Set2
    for i, metric in enumerate(metrics_list, 1):
        for j, model_name in enumerate(model_names):
            val = perf_df[perf_df['Model'] == model_name][metric].values[0]
            fig_perf.add_trace(
                go.Bar(
                    name=model_name if i == 1 else None,
                    x=[model_name], y=[val],
                    marker_color=colors[j % len(colors)],
                    text=[f"{val:.4f}"], textposition='outside',
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
        fig_perf.update_yaxes(range=[0.5, 1.05], row=1, col=i)
    fig_perf.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_perf, use_container_width=True)

with tab2:
    st.dataframe(
        perf_df.style.highlight_max(subset=['Test Accuracy', 'Test F1'], color='#C8E6C9').format(precision=4),
        use_container_width=True
    )

# ============================================================
# 2. Confusion Matrix
# ============================================================
st.markdown("## 2️⃣ Confusion Matrix")

cm_cols = st.columns(min(len(model_names), 4))
color_scales = ['Blues', 'Purples', 'Greens', 'Oranges']

for idx, name in enumerate(model_names):
    cm = np.array(metrics_data[name].get('confusion_matrix', []))
    if cm.size == 0:
        continue
    col_idx = idx % len(cm_cols)
    with cm_cols[col_idx]:
        st.markdown(f"### {name}")
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale=color_scales[idx % len(color_scales)],
            labels={'x': '예측', 'y': '실제'}
        )
        fig_cm.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

# ============================================================
# 3. 클래스별 상세 성능
# ============================================================
st.markdown("## 3️⃣ 클래스별 상세 성능")

selected_model = st.selectbox("모델 선택", model_names)
report = metrics_data[selected_model].get('classification_report', {})

class_rows = []
for label in labels:
    if label in report:
        class_rows.append({
            '등급': label,
            'Precision': report[label].get('precision', 0),
            'Recall': report[label].get('recall', 0),
            'F1-Score': report[label].get('f1-score', 0),
            'Support': int(report[label].get('support', 0))
        })

if class_rows:
    class_df = pd.DataFrame(class_rows)

    fig_class = go.Figure()
    for metric in ['Precision', 'Recall', 'F1-Score']:
        fig_class.add_trace(go.Bar(
            name=metric, x=class_df['등급'], y=class_df[metric],
            text=class_df[metric].round(3), textposition='outside'
        ))
    fig_class.update_layout(
        barmode='group', height=400, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_range=[0, 1.1], xaxis_title="불평등 등급", yaxis_title="Score"
    )
    st.plotly_chart(fig_class, use_container_width=True)
    st.dataframe(class_df.style.format(precision=4), use_container_width=True)

# ============================================================
# 4. 전체 평균 지표
# ============================================================
st.markdown("## 4️⃣ 전체 평균 지표")

avg_cols = st.columns(3)
for avg_type, col in zip(['macro avg', 'weighted avg', 'accuracy'], avg_cols):
    with col:
        if avg_type == 'accuracy' and 'accuracy' in report:
            st.metric("Accuracy", f"{report['accuracy']:.4f}")
        elif avg_type in report:
            st.metric(
                avg_type.replace('avg', '평균').title(),
                f"F1: {report[avg_type].get('f1-score', 0):.4f}"
            )
            st.caption(
                f"Precision: {report[avg_type].get('precision', 0):.4f} / "
                f"Recall: {report[avg_type].get('recall', 0):.4f}"
            )