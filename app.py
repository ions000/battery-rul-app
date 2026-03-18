import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import zipfile
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH Analyzer", layout="wide")
st.title("🔋 사용자 맞춤형 배터리 SOH 분석 플랫폼")
st.markdown("학습된 **모델 패키지(ZIP)**와 **데이터(CSV)**를 업로드하여 즉시 결과를 확인하세요.")

# 2. 모델 로드 함수 (ZIP 내 .keras와 .pkl 추출)
def load_model_package(zip_file):
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            # .keras 모델 파일 찾기
            model_files = [f for f in z.namelist() if f.endswith('.keras')]
            # .pkl 스케일러 파일 찾기
            scaler_files = [f for f in z.namelist() if f.endswith('.pkl')]
            
            if not model_files or not scaler_files:
                return None, None, "ZIP 파일 내에 .keras 모델 또는 .pkl 스케일러가 없습니다."
            
            # 모델 로드 (임시 파일 저장 후 로드)
            z.extract(model_files[0], "temp_dir")
            model = load_model(f"temp_dir/{model_files[0]}")
            
            # 스케일러 로드
            with z.open(scaler_files[0]) as f:
                sc_X, sc_y = pickle.load(f)
                
            return model, (sc_X, sc_y), None
    except Exception as e:
        return None, None, str(e)

# 3. 데이터 준비 함수
def prepare_test_data(df, test_ids, seq_length, sc_X, sc_y):
    features, target = ['voltage', 'temperature', 'capacity'], 'soh'
    X_list, y_list, cycle_list = [], [], []
    
    for b_id in test_ids:
        b_df = df[df['battery_id'] == b_id].sort_values('cycle')
        if len(b_df) <= seq_length: continue
        
        f_s = sc_X.transform(b_df[features])
        t_s = sc_y.transform(b_df[[target]])
        
        for i in range(len(f_s) - seq_length):
            X_list.append(f_s[i:i+seq_length])
            y_list.append(t_s[i+seq_length])
            cycle_list.append(b_df['cycle'].iloc[i+seq_length])
            
    return np.array(X_list).astype(np.float32), np.array(y_list).astype(np.float32), np.array(cycle_list)

# 4. 사이드바 - 파일 업로드 센터
st.sidebar.header("📁 업로드 센터")
uploaded_model_zip = st.sidebar.file_uploader("1️⃣ 모델 패키지(ZIP) 업로드", type="zip")
uploaded_data_csv = st.sidebar.file_uploader("2️⃣ 배터리 데이터(CSV) 업로드", type="csv")

st.sidebar.markdown("---")
window_size = st.sidebar.slider("Window Size (학습 시 설정값)", 1, 50, 6)

# 5. 메인 분석 로직
if uploaded_model_zip and uploaded_data_csv:
    # 데이터 먼저 읽기
    df = pd.read_csv(uploaded_data_csv)
    all_ids = sorted(df['battery_id'].unique())
    
    st.subheader("🎯 분석 대상 설정")
    test_ids = st.multiselect("예측할 배터리 ID를 선택하세요", all_ids, default=[all_ids[-1]])

    if st.button("🚀 분석 실행"):
        with st.spinner("모델 패키지를 해제하고 예측을 수행 중입니다..."):
            # 모델 패키지 로드
            model, scalers, error = load_model_package(uploaded_model_zip)
            
            if error:
                st.error(f"모델 로드 실패: {error}")
            else:
                sc_X, sc_y = scalers
                # 테스트 데이터 준비
                X_test, y_test, cycles = prepare_test_data(df, test_ids, window_size, sc_X, sc_y)
                
                if len(X_test) == 0:
                    st.warning("데이터가 부족하여 분석할 수 없습니다. Window Size를 조절해 보세요.")
                else:
                    # 예측
                    y_pred_scaled = model.predict(X_test)
                    y_pred = sc_y.inverse_transform(y_pred_scaled).flatten()
                    y_actual = sc_y.inverse_transform(y_test).flatten()
                    
                    # 결과 전시
                    st.success("✅ 분석 완료!")
                    
                    col_l, col_r = st.columns([7, 3])
                    with col_l:
                        st.subheader("📈 SOH 추종 결과")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(cycles, y_actual, label='Actual SOH', color='#3498db', linewidth=2)
                        ax.plot(cycles, y_pred, label='Predicted SOH', color='#e67e22', linestyle='--')
                        ax.set_xlabel("Cycle"); ax.set_ylabel("SOH"); ax.legend(); ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col_r:
                        st.subheader("📋 수치 데이터")
                        res_df = pd.DataFrame({
                            'Cycle': cycles.astype(int),
                            'Actual': np.round(y_actual, 4),
                            'Predicted': np.round(y_pred, 4)
                        })
                        st.dataframe(res_df.head(50), use_container_width=True)
                        st.metric("평균 절대 오차 (MAE)", f"{mean_absolute_error(y_actual, y_pred):.5f}")

elif not uploaded_model_zip:
    st.info("👈 왼쪽 사이드바에서 **모델 패키지(ZIP)**를 먼저 업로드해 주세요.")
else:
    st.info("👈 왼쪽 사이드바에서 분석할 **배터리 데이터(CSV)**를 업로드해 주세요.")
