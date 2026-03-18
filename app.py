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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH Analyzer Pro", layout="wide")
st.title("🔋 배터리 SOH 통합 분석 플랫폼")

# --- [설정] 기본 모델 확인 ---
DEFAULT_ZIP = "battery_package.zip"
has_default = os.path.exists(DEFAULT_ZIP)

# 2. 모델 로드 함수 (ZIP)
def load_model_package(zip_source):
    try:
        z = zipfile.ZipFile(zip_source, 'r')
        with z:
            model_files = [f for f in z.namelist() if f.endswith('.keras')]
            scaler_files = [f for f in z.namelist() if f.endswith('.pkl')]
            if not model_files or not scaler_files:
                return None, None, "ZIP 내부에 .keras 또는 .pkl이 없습니다."
            z.extract(model_files[0], "temp_dir")
            model = load_model(f"temp_dir/{model_files[0]}")
            with z.open(scaler_files[0]) as f:
                sc_X, sc_y = pickle.load(f)
            return model, (sc_X, sc_y), None
    except Exception as e:
        return None, None, str(e)

# 3. 데이터 준비 함수
def prepare_data(df, train_ids, test_ids, seq_length, sc_X=None, sc_y=None):
    features, target = ['voltage', 'temperature', 'capacity'], 'soh'
    if sc_X is None: # 신규 학습 시 스케일러 생성
        sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
        train_df = df[df['battery_id'].isin(train_ids)]
        if train_df.empty: return None, None, None, None, None, None, None
        sc_X.fit(train_df[features]); sc_y.fit(train_df[[target]])

    def create_seq(target_ids):
        X_list, y_list, cycle_list = [], [], []
        for b_id in target_ids:
            b_df = df[df['battery_id'] == b_id].sort_values('cycle')
            if len(b_df) <= seq_length: continue
            f_s, t_s = sc_X.transform(b_df[features]), sc_y.transform(b_df[[target]])
            for i in range(len(f_s) - seq_length):
                X_list.append(f_s[i:i+seq_length]); y_list.append(t_s[i+seq_length])
                cycle_list.append(b_df['cycle'].iloc[i+seq_length])
        return np.array(X_list).astype(np.float32), np.array(y_list).astype(np.float32), np.array(cycle_list)

    X_tr, y_tr, _ = create_seq(train_ids)
    X_te, y_te, cycles = create_seq(test_ids)
    return X_tr, y_tr, X_te, y_te, cycles, sc_X, sc_y

# 4. 사이드바 설정
st.sidebar.header("📂 모드 설정")
analysis_mode = st.sidebar.radio("작동 모드", ["기본/업로드 모델 사용", "신규 모델 직접 학습"])

st.sidebar.markdown("---")
window_size = st.sidebar.slider("Window Size", 1, 50, 6)

if analysis_mode == "신규 모델 직접 학습":
    st.sidebar.subheader("🚀 학습 파라미터")
    epochs = st.sidebar.number_input("Epochs", 10, 500, 150)
    batch_size = st.sidebar.number_input("Batch Size", 1, 128, 40)
else:
    uploaded_zip = st.sidebar.file_uploader("커스텀 ZIP 업로드 (미업로드 시 기본모델 사용)", type="zip")

# 5. 메인 화면 로직
uploaded_csv = st.file_uploader("CSV 데이터 업로드", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    all_ids = sorted(df['battery_id'].unique())
    test_ids = st.multiselect("🎯 예측 대상 배터리 ID (Test)", all_ids, default=[all_ids[-1]])

    # --- 모드 1: 기존 모델 활용 ---
    if analysis_mode == "기본/업로드 모델 사용":
        model_source = uploaded_zip if uploaded_zip else (DEFAULT_ZIP if has_default else None)
        
        if model_source and st.button("🔌 즉시 분석 실행"):
            model, scalers, err = load_model_package(model_source)
            if err: st.error(err)
            else:
                _, _, X_te, y_te, cycles, sc_X, sc_y = prepare_data(df, [], test_ids, window_size, scalers[0], scalers[1])
                y_p = sc_y.inverse_transform(model.predict(X_te)).flatten()
                y_a = sc_y.inverse_transform(y_te).flatten()
                st.session_state['res'] = (y_a, y_p, cycles)

    # --- 모드 2: 신규 학습 및 다운로드 ---
    else:
        train_ids = st.multiselect("📚 학습용 배터리 ID (Train)", all_ids, default=[i for i in all_ids if i not in test_ids])
        if st.button("🚀 학습 시작 및 다운로드 생성"):
            with st.spinner("모델 학습 중..."):
                X_tr, y_tr, X_te, y_te, cycles, sc_X, sc_y = prepare_data(df, train_ids, test_ids, window_size)
                
                # 모델 구축 (Bi-LSTM)
                model = Sequential([
                    Input(shape=(window_size, X_tr.shape[2])),
                    Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
                    Dropout(0.1),
                    Bidirectional(LSTM(64, activation='tanh')),
                    Dense(32, activation='relu'), Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                
                bar = st.progress(0)
                model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0,
                          callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e,l: bar.progress((e+1)/epochs))])
                
                y_p = sc_y.inverse_transform(model.predict(X_te)).flatten()
                y_a = sc_y.inverse_transform(y_te).flatten()
                st.session_state['res'] = (y_a, y_p, cycles)

                # --- [추가] 학습 완료 후 ZIP 다운로드 버튼 생성 ---
                model.save("new_model.keras")
                with open("new_sc.pkl", "wb") as f: pickle.dump((sc_X, sc_y), f)
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as z:
                    z.write("new_model.keras"); z.write("new_sc.pkl")
                
                st.success("🎉 학습이 완료되었습니다! 아래 버튼으로 모델을 저장하세요.")
                st.download_button("📥 학습된 모델 패키지(ZIP) 다운로드", zip_buf.getvalue(), "battery_package.zip")

    # 6. 결과 출력
    if 'res' in st.session_state:
        y_actual, y_pred, cycles = st.session_state['res']
        st.divider()
        c1, c2 = st.columns([7, 3])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cycles, y_actual, label='Actual', color='#3498db')
            ax.plot(cycles, y_pred, label='Predicted', color='#e67e22', linestyle='--')
            ax.legend(); st.pyplot(fig)
        with c2:
            st.metric("MAE", f"{mean_absolute_error(y_actual, y_pred):.5f}")
            st.dataframe(pd.DataFrame({'Cycle': cycles, 'Actual': y_actual, 'Pred': y_pred}).head(20))
