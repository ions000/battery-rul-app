import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH Master", layout="wide")
st.title("🔋 배터리 SOH 정밀 예측 및 모델 관리 플랫폼")

# 2. 사이드바 - 모델 관리 (저장된 모델 불러오기)
st.sidebar.header("📁 모델 재사용 (선택사항)")
uploaded_model = st.sidebar.file_uploader("학습된 .keras 모델 업로드", type="keras")
uploaded_scaler = st.sidebar.file_uploader("저장된 스케일러(.pkl) 업로드", type="pkl")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 신규 학습 설정")
WINDOW_SIZE = st.sidebar.slider("Window Size", 1, 50, 6)
EPOCHS = st.sidebar.number_input("Epochs", 1, 500, 150)
BATCH_SIZE = st.sidebar.number_input("Batch Size", 1, 128, 40)

# 3. 데이터 준비 함수
def prepare_data(df, train_ids, test_ids, seq_length, scaler_X=None, scaler_y=None):
    features = ['voltage', 'temperature', 'capacity']
    target = 'soh'
    
    if scaler_X is None or scaler_y is None:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        train_df = df[df['battery_id'].isin(train_ids)]
        scaler_X.fit(train_df[features])
        scaler_y.fit(train_df[[target]])

    def create_seq(target_ids):
        X_list, y_list, cycle_list = [], [], []
        for b_id in target_ids:
            b_df = df[df['battery_id'] == b_id].sort_values('cycle')
            if len(b_df) <= seq_length: continue
            f_scaled = scaler_X.transform(b_df[features])
            t_scaled = scaler_y.transform(b_df[[target]])
            for i in range(len(f_scaled) - seq_length):
                X_list.append(f_scaled[i:i+seq_length])
                y_list.append(t_scaled[i+seq_length])
                cycle_list.append(b_df['cycle'].iloc[i+seq_length])
        return np.array(X_list).astype(np.float32), np.array(y_list).astype(np.float32), np.array(cycle_list)

    X_train, y_train, _ = create_seq(train_ids)
    X_test, y_test, test_cycles = create_seq(test_ids)
    return X_train, y_train, X_test, y_test, test_cycles, scaler_X, scaler_y

# 4. 파일 업로드
uploaded_csv = st.file_uploader("분석할 NASA 배터리 CSV 업로드", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    all_ids = sorted(df['battery_id'].unique())
    train_ids = st.sidebar.multiselect("학습용 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트용 ID", all_ids, default=[all_ids[-1]])

    # 상황 1: 저장된 모델과 스케일러가 모두 있을 때 (바로 예측)
    if uploaded_model and uploaded_scaler:
        if st.button("🔌 업로드된 모델로 1초 만에 예측"):
            with open("temp_model.keras", "wb") as f: f.write(uploaded_model.getbuffer())
            model = load_model("temp_model.keras")
            sc_X, sc_y = pickle.load(uploaded_scaler)
            
            _, _, X_test, y_test, test_cycles, _, _ = prepare_data(df, train_ids, test_ids, WINDOW_SIZE, sc_X, sc_y)
            
            y_pred = sc_y.inverse_transform(model.predict(X_test)).flatten()
            y_actual = sc_y.inverse_transform(y_test).flatten()
            
            # (결과 시각화 로직 동일 - 하단 참조)
            st.success("✅ 기존 모델 로드 성공!")

    # 상황 2: 새로 학습할 때
    if st.button("🚀 신규 Bi-LSTM 학습 시작"):
        X_train, y_train, X_test, y_test, test_cycles, sc_X, sc_y = prepare_data(df, train_ids, test_ids, WINDOW_SIZE)
        
        model = Sequential([
            Input(shape=(WINDOW_SIZE, X_train.shape[2])),
            Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
            Dropout(0.1),
            Bidirectional(LSTM(64, activation='tanh')),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        bar = st.progress(0)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
                  callbacks=[early_stop, tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e,l: bar.progress((e+1)/EPOCHS))], verbose=0)
        
        y_pred = sc_y.inverse_transform(model.predict(X_test)).flatten()
        y_actual = sc_y.inverse_transform(y_test).flatten()

        # 결과 출력 및 모델 다운로드 버튼
        st.success("✅ 학습 완료!")
        
        # 모델 저장용
        model.save("trained_model.keras")
        with open("trained_model.keras", "rb") as f:
            st.download_button("📥 모델(.keras) 다운로드", f, "battery_model.keras")
        
        # 스케일러 저장용 (pickle)
        scalers = (sc_X, sc_y)
        with open("scalers.pkl", "wb") as f: pickle.dump(scalers, f)
        with open("scalers.pkl", "rb") as f:
            st.download_button("📥 스케일러(.pkl) 다운로드", f, "scalers.pkl")

    # 결과 시각화 (y_actual, y_pred가 정의된 경우에만 실행)
    if 'y_actual' in locals():
        col_l, col_r = st.columns([7, 3])
        with col_l:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(test_cycles, y_actual, label='Actual', color='#3498db')
            ax.plot(test_cycles, y_pred, label='Predicted', color='#e67e22', linestyle='--')
            ax.legend(); st.pyplot(fig)
        with col_r:
            res_df = pd.DataFrame({'Cycle': test_cycles.astype(int), 'Actual': y_actual, 'Predicted': y_pred})
            st.dataframe(res_df.head(20))
            st.metric("MAE", f"{mean_absolute_error(y_actual, y_pred):.5f}")
