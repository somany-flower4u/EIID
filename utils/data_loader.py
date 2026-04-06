import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 피처 정의 (기존 파이프라인과 동일)
# ============================================================
NUMERIC_FEATURES_DEFAULT = [
    '급식_전체학생수', '급식_급식비율', '급식_인력당학생수',
    '학생1인당_수입', '학생1인당_전체공간', '학생1인당_교실',
    '학생1인당_화장실', '학생백명당_방과후프로그램', '방과후_미운영'
]

CATEGORICAL_FEATURES_DEFAULT = [
    '학교급', '설립구분', '급식_학교규모',
    '급식_급식참여등급', '급식_영양교사배치', '급식_인력적정성'
]

# ============================================================
# 캐싱 로드 함수들
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv('data/merge_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_model.pkl')

@st.cache_resource
def load_label_encoder():
    return joblib.load('models/label_encoder.pkl')

@st.cache_data
def load_metrics():
    path = Path('models/metrics.pkl')
    return joblib.load(path) if path.exists() else None

@st.cache_data
def load_shap_data():
    path = Path('models/shap_data.pkl')
    return joblib.load(path) if path.exists() else None

@st.cache_data
def load_feature_columns():
    path = Path('models/feature_columns.pkl')
    return joblib.load(path) if path.exists() else None

@st.cache_data
def build_feature_matrix(_df, numeric_features, categorical_features):
    """모델 입력용 피처 매트릭스 생성 (기존 파이프라인과 동일 로직)"""
    target_col = '교육불평등_등급'
    cols_needed = numeric_features + categorical_features + [target_col]
    cols_exist = [c for c in cols_needed if c in _df.columns]

    df_model = _df[cols_exist].copy()
    df_model = df_model.dropna(subset=[target_col]) if target_col in df_model.columns else df_model

    cat_exist = [c for c in categorical_features if c in df_model.columns]
    df_encoded = pd.get_dummies(df_model, columns=cat_exist, drop_first=True)

    X = df_encoded.drop(target_col, axis=1, errors='ignore')
    y = df_encoded[target_col] if target_col in df_encoded.columns else None
    X = X.fillna(X.median())
    return X, y

# ============================================================
# 통합 초기화 함수 (모든 페이지에서 호출)
# ============================================================
def init_session():
    """
    어떤 페이지에서든 이 함수만 호출하면
    df, model, le, X, y 등 모든 공유 객체가 session_state에 준비됩니다.
    이미 로드된 경우 캐시 덕분에 즉시 반환됩니다.
    """
    # --- 데이터 ---
    if 'df' not in st.session_state:
        st.session_state['df'] = load_data()

    df = st.session_state['df']

    # --- 모델 / 인코더 ---
    if 'model' not in st.session_state:
        st.session_state['model'] = load_model()
    if 'le' not in st.session_state:
        st.session_state['le'] = load_label_encoder()

    # --- 피처 목록 (실제 컬럼 존재 여부 필터) ---
    if 'NUMERIC_FEATURES' not in st.session_state:
        st.session_state['NUMERIC_FEATURES'] = [
            c for c in NUMERIC_FEATURES_DEFAULT if c in df.columns
        ]
    if 'CATEGORICAL_FEATURES' not in st.session_state:
        st.session_state['CATEGORICAL_FEATURES'] = [
            c for c in CATEGORICAL_FEATURES_DEFAULT if c in df.columns
        ]

    # --- 피처 매트릭스 ---
    if 'X' not in st.session_state:
        X, y = build_feature_matrix(
            df,
            st.session_state['NUMERIC_FEATURES'],
            st.session_state['CATEGORICAL_FEATURES']
        )

        # 저장된 피처 컬럼 순서에 맞춤
        saved_cols = load_feature_columns()
        if saved_cols is not None:
            for col in saved_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[saved_cols]

        st.session_state['X'] = X
        st.session_state['y'] = y

    # --- 메트릭 / SHAP ---
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = load_metrics()
    if 'shap_data' not in st.session_state:
        st.session_state['shap_data'] = load_shap_data()

    return {
        'df': st.session_state['df'],
        'model': st.session_state['model'],
        'le': st.session_state['le'],
        'X': st.session_state['X'],
        'y': st.session_state['y'],
        'NUMERIC_FEATURES': st.session_state['NUMERIC_FEATURES'],
        'CATEGORICAL_FEATURES': st.session_state['CATEGORICAL_FEATURES'],
        'metrics': st.session_state['metrics'],
        'shap_data': st.session_state['shap_data'],
    }
