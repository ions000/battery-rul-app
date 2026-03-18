import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH & RUL Lab", layout="wide")
st.title("🔋 NASA 배터리 SOH 퇴화 및 RUL 예측 시스템")
st.markdown("""
이 시스템은 배터리의 **건강 상태(SOH)** 변화를 관찰하고, 이를 바탕으로 **남은 수명(RUL)**을 예측합니다.
""")

# 2. 사이드바 설정
st.sidebar.header("⚙️ 학습 설정")
window_size = st.sidebar.slider("Window Size (과거 참조 사이클)", 5, 50, 10)
epochs = st.sidebar.number_input("Epochs (학습 횟수)", 10, 100, 30)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# 3. 데이터 전처리 함수
def create_sequences_by_id(df, selected_ids, window, scaler=None, is_train=True):
    X, y = [], []
    features = ['voltage', 'temperature', 'capacity', 'soh']
    
    if is_train:
        scaler = MinMaxScaler()
        scaler.fit(df[df['battery_id'].isin(selected_ids)][features])
    
    for b_id in selected_ids:
        temp_df = df[df['battery_id'] == b_id].sort_values('cycle')
        scaled_features = scaler.transform(temp_df[features])
        target_values = temp_df['rul'].values
        
        for i in range(len(temp_df) - window):
            X.append(scaled_features[i:i+window])
            y.append(target_values[i+window])
            
    return np.array(X), np.array(y), scaler

# 4. 파일 업로드
uploaded_file = st.file_uploader("NASA 배터리 CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_ids = sorted(df['battery_id'].unique())
    
    st.sidebar.markdown("---")
    st.sidebar.header("📂 데이터 분할")
    train_ids = st.sidebar.multiselect("학습용 배터리 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트용 배터리 ID", all_ids, default=[all_ids[-1]])

    if st.button("🚀 분석 및 학습 시작"):
        if not train_ids or not test_ids:
            st.error("⚠️ 학습용과 테스트용 배터리를 각각 선택해주세요.")
        else:
            # 데이터 준비
            X_train, y_train, scaler = create_sequences_by_id(df, train_ids, window_size, is_train=True)
            X_test, y_test, _ = create_sequences_by_id(df, test_ids, window_size, scaler=scaler, is_train=False)

            # 5. 모델 학습 (Bidirectional LSTM)
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.2),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            progress_bar = st.progress(0)
            status_text = st.empty()

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f}")

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      validation_split=0.1, callbacks=[early_stop, StreamlitCallback()], verbose=0)

            # 6. 결과 시각화 섹션
            st.success("✅ 학습 및 예측이 완료되었습니다!")
            
            # --- 그래프 레이아웃 설정 ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📉 SOH 퇴화 추이 (실제 데이터)")
                fig_soh, ax_soh = plt.subplots()
                for b_id in test_ids:
                    soh_vals = df[df['battery_id'] == b_id].sort_values('cycle')['soh']
                    ax_soh.plot(soh_vals.values, label=f'Battery {b_id}')
                ax_soh.set_xlabel('Cycle')
                ax_soh.set_ylabel('SOH')
                ax_soh.axhline(y=0.8, color='r', linestyle='--', label='Threshold (0.8)')
                ax_soh.legend()
                st.pyplot(fig_soh)

            with col2:
                st.subheader("📈 RUL 예측 결과 (모델 출력)")
                y_pred = model.predict(X_test)
                fig_rul, ax_rul = plt.subplots()
                ax_rul.plot(y_test, label='Actual RUL', color='green')
                ax_rul.plot(y_pred, label='Predicted RUL', color='red', linestyle='--')
                ax_rul.set_xlabel('Time Step')
                ax_rul.set_ylabel('Remaining Cycles')
                ax_rul.legend()
                st.pyplot(fig_rul)

            # 성능 지표
            mae = mean_absolute_error(y_test, y_pred)
            st.info(f"💡 테스트 배터리({test_ids})에 대한 평균 오차(MAE): **{mae:.2f} 사이클**")
else:
    st.info("파일을 업로드하면 분석이 시작됩니다.")
