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
st.set_page_config(page_title="Battery SOH AI Lab", layout="wide")
st.title("🔋 배터리 SOH 정밀 분석 플랫폼")

# --- [설정] 서버에 기본 모델 패키지 확인 ---
DEFAULT_ZIP = "battery_package.zip"
has_default = os.path.exists(DEFAULT_ZIP)

# 2. 모델 로드 함수 (ZIP 압축 해제)
def load_model_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        model_name = [f for f in z.namelist() if f.endswith('.keras')][0]
        z.extract(model_name)
        model = load_model(model_name)
        scaler_name = [f for f in z.namelist() if f.endswith('.pkl')][0]
        with z.open(scaler_name) as f:
            sc_X, sc_y = pickle.load(f)
    return model, sc_X, sc_y

# 3. 데이터 준비 함수
def prepare_data(df, train_ids, test_ids, seq_length, sc_X=None, sc_y=None):
    features, target = ['voltage', 'temperature', 'capacity'], 'soh'
    if sc_X is None:
        sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
        train_df = df[df['battery_id'].isin(train_ids)]
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

    X_train, y_train, _ = create_seq(train_ids)
    X_test, y_test, test_cycles = create_seq(test_ids)
    return X_train, y_train, X_test, y_test, test_cycles, sc_X, sc_y

# 4. 사이드바 설정
st.sidebar.header("⚙️ 분석 설정")
if has_default:
    st.sidebar.success("📦 기본 모델 패키지 감지됨")
    mode = st.sidebar.radio("모드 선택", ["기본 모델 사용", "신규 학습"])
else:
    st.sidebar.warning("⚠️ 기본 모델 없음 (신규 학습 필요)")
    mode = "신규 학습"

window_size = st.sidebar.slider("Window Size", 1, 50, 6)
if mode == "신규 학습":
    epochs = st.sidebar.number_input("Epochs", 10, 500, 150)
    batch_size = st.sidebar.number_input("Batch Size", 1, 128, 40)

# 5. 파일 업로드 및 메인 로직
uploaded_csv = st.file_uploader("NASA 배터리 CSV 데이터를 업로드하세요", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    all_ids = sorted(df['battery_id'].unique())
    test_ids = st.sidebar.multiselect("예측 대상 배터리 ID", all_ids, default=[all_ids[-1]])

    if mode == "기본 모델 사용":
        if st.button("🔌 기본 모델로 즉시 분석"):
            model, sc_X, sc_y = load_model_from_zip(DEFAULT_ZIP)
            _, _, X_test, y_test, cycles, _, _ = prepare_data(df, [], test_ids, window_size, sc_X, sc_y)
            y_pred = sc_y.inverse_transform(model.predict(X_test)).flatten()
            y_actual = sc_y.inverse_transform(y_test).flatten()
            st.session_state['res'] = (y_actual, y_pred, cycles)

    else: # 신규 학습 모드 (기존 화면과 동일)
        if st.button("🚀 신규 Bi-LSTM 학습 시작"):
            train_ids = [i for i in all_ids if i not in test_ids]
            X_tr, y_tr, X_te, y_te, cycles, sc_X, sc_y = prepare_data(df, train_ids, test_ids, window_size)
            
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
            
            y_pred = sc_y.inverse_transform(model.predict(X_te)).flatten()
            y_actual = sc_y.inverse_transform(y_te).flatten()
            st.session_state['res'] = (y_actual, y_pred, cycles)

            # 새 모델 ZIP 저장 기능
            model.save("new_model.keras")
            with open("new_scalers.pkl", "wb") as f: pickle.dump((sc_X, sc_y), f)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as z:
                z.write("new_model.keras"); z.write("new_scalers.pkl")
            st.download_button("📥 현재 모델 패키지(ZIP) 다운로드", zip_buf.getvalue(), "battery_package.zip")

    # 6. 결과 출력 (공통)
    if 'res' in st.session_state:
        y_a, y_p, cyc = st.session_state['res']
        st.success(f"✅ 분석 완료! (MAE: {mean_absolute_error(y_a, y_p):.5f})")
        
        col_l, col_r = st.columns([7, 3])
        with col_l:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cyc, y_a, label='Actual', color='#3498db', linewidth=2)
            ax.plot(cyc, y_p, label='Predicted', color='#e67e22', linestyle='--')
            ax.set_xlabel("Cycle"); ax.set_ylabel("SOH"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        with col_r:
            st.write("📋 상세 수치")
            st.dataframe(pd.DataFrame({'Cycle': cyc.astype(int), 'Actual': np.round(y_a, 4), 'Pred': np.round(y_p, 4)}).head(30))
else:
    st.info("💡 CSV 데이터를 업로드하면 분석이 시작됩니다.")
