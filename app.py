import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
import zipfile
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH Analyzer Pro", layout="wide")
st.title("🔋 배터리 SOH 지능형 분석 플랫폼")
st.markdown("기본 모델 또는 업로드한 모델 패키지(ZIP)를 사용하여 SOH를 예측합니다.")

# --- [설정] 서버 내 기본 모델 경로 ---
DEFAULT_ZIP = "battery_package.zip"
has_default = os.path.exists(DEFAULT_ZIP)

# 2. 모델 로드 함수 (ZIP 압축 해제 및 검증)
def load_model_package(zip_file_source):
    """
    zip_file_source: 파일 경로(str) 또는 UploadedFile 객체
    """
    try:
        # 파일 경로인 경우와 업로드된 객체인 경우 구분하여 처리
        if isinstance(zip_file_source, str):
            z = zipfile.ZipFile(zip_file_source, 'r')
        else:
            z = zipfile.ZipFile(zip_file_source, 'r')
            
        with z:
            model_files = [f for f in z.namelist() if f.endswith('.keras')]
            scaler_files = [f for f in z.namelist() if f.endswith('.pkl')]
            
            if not model_files or not scaler_files:
                return None, None, "ZIP 내부에 .keras 모델 또는 .pkl 스케일러가 없습니다."
            
            # 모델 로드 (임시 추출 필요)
            z.extract(model_files[0], "temp_dir")
            model = load_model(f"temp_dir/{model_files[0]}")
            
            # 스케일러 로드
            with z.open(scaler_files[0]) as f:
                sc_X, sc_y = pickle.load(f)
                
            return model, (sc_X, sc_y), None
    except Exception as e:
        return None, None, str(e)

# 3. 데이터 준비 함수 (SOH 전용)
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

# 4. 사이드바 - 모델 및 데이터 설정
st.sidebar.header("📁 데이터 및 모델 설정")

# 4-1. 모델 선택 로직
st.sidebar.subheader("🤖 모델 소스")
uploaded_zip = st.sidebar.file_uploader("커스텀 모델(ZIP) 업로드 (선택)", type="zip")

if uploaded_zip:
    st.sidebar.info("✨ 업로드된 커스텀 모델을 사용합니다.")
    model_source = uploaded_zip
elif has_default:
    st.sidebar.success("📦 GitHub 기본 모델을 사용합니다.")
    model_source = DEFAULT_ZIP
else:
    st.sidebar.error("❌ 사용 가능한 모델이 없습니다. ZIP을 업로드하세요.")
    model_source = None

st.sidebar.markdown("---")
# 4-2. 데이터 업로드
uploaded_csv = st.sidebar.file_uploader("분석할 CSV 데이터 업로드", type="csv")
window_size = st.sidebar.slider("Window Size (Sequence Length)", 1, 50, 6)

# 5. 메인 로직 실행
if uploaded_csv and model_source:
    df = pd.read_csv(uploaded_csv)
    all_ids = sorted(df['battery_id'].unique())
    
    st.subheader("🎯 분석 대상 설정")
    test_ids = st.multiselect("예측할 배터리 ID를 선택하세요", all_ids, default=[all_ids[-1]])

    if st.button("🚀 분석 실행"):
        with st.spinner("모델 로딩 및 분석 중..."):
            model, scalers, error = load_model_package(model_source)
            
            if error:
                st.error(f"모델 로드 오류: {error}")
            else:
                sc_X, sc_y = scalers
                X_test, y_test, cycles = prepare_test_data(df, test_ids, window_size, sc_X, sc_y)
                
                if len(X_test) == 0:
                    st.warning("데이터가 부족하여 시퀀스를 생성할 수 없습니다. Window Size를 낮춰보세요.")
                else:
                    # 예측 및 역스케일링
                    y_pred_scaled = model.predict(X_test)
                    y_pred = sc_y.inverse_transform(y_pred_scaled).flatten()
                    y_actual = sc_y.inverse_transform(y_test).flatten()
                    
                    st.success("✅ 분석이 완료되었습니다!")
                    
                    # 6. 결과 시각화
                    col_l, col_r = st.columns([7, 3])
                    with col_l:
                        st.subheader("📈 SOH 예측 결과 그래프")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(cycles, y_actual, label='Actual SOH', color='#3498db', linewidth=2)
                        ax.plot(cycles, y_pred, label='Predicted SOH', color='#e67e22', linestyle='--')
                        ax.set_xlabel("Cycle")
                        ax.set_ylabel("SOH (State of Health)")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col_r:
                        st.subheader("📋 수치 데이터 요약")
                        res_df = pd.DataFrame({
                            'Cycle': cycles.astype(int),
                            'Actual': np.round(y_actual, 4),
                            'Predicted': np.round(y_pred, 4)
                        })
                        st.dataframe(res_df.head(50), use_container_width=True)
                        
                        mae = mean_absolute_error(y_actual, y_pred)
                        st.metric("평균 절대 오차 (MAE)", f"{mae:.5f}")
                        
                        # 결과 다운로드
                        csv_data = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 결과 CSV 다운로드", csv_data, "prediction_results.csv", "text/csv")
else:
    # 초기 화면 안내
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 왼쪽 사이드바에서 분석할 **CSV 데이터**를 업로드해 주세요.")
    with col2:
        if not has_default and not uploaded_zip:
            st.warning("⚠️ 현재 서버에 기본 모델이 없습니다. GitHub에 `battery_package.zip`을 추가하거나 파일을 업로드하세요.")
        elif uploaded_zip:
            st.success("✅ 커스텀 모델이 준비되었습니다.")
        else:
            st.success("✅ 기본 모델(`battery_package.zip`)이 준비되었습니다.")
