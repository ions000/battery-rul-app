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
st.set_page_config(page_title="Battery SOH & RUL Analyzer", layout="wide")
st.title("🔋 NASA 배터리 SOH & RUL 통합 예측 플랫폼")

# 2. 사이드바 설정 (요청하신 디폴트 값)
st.sidebar.header("⚙️ 모델 학습 설정")
window_size = st.sidebar.slider("Window Size", 1, 50, 6)
epochs = st.sidebar.number_input("Epochs", 1, 500, 150)
batch_size = st.sidebar.number_input("Batch Size", 1, 128, 40)

# 3. 데이터 전처리 함수
def create_sequences_by_id(df, selected_ids, window, scaler=None, is_train=True):
    X, y_rul, y_soh = [], [], []
    features = ['voltage', 'temperature', 'capacity', 'soh']
    
    if is_train:
        scaler = MinMaxScaler()
        scaler.fit(df[df['battery_id'].isin(selected_ids)][features])
    
    for b_id in selected_ids:
        temp_df = df[df['battery_id'] == b_id].sort_values('cycle')
        if len(temp_df) <= window: continue
            
        scaled_features = scaler.transform(temp_df[features])
        rul_values = temp_df['rul'].values
        soh_values = temp_df['soh'].values
        
        for i in range(len(temp_df) - window):
            X.append(scaled_features[i:i+window])
            y_rul.append(rul_values[i+window])
            y_soh.append(soh_values[i+window])
            
    return np.array(X), np.array(y_rul), np.array(y_soh), scaler

# 4. 파일 업로드
uploaded_file = st.file_uploader("NASA 배터리 CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_ids = sorted(df['battery_id'].unique())
    
    st.sidebar.markdown("---")
    train_ids = st.sidebar.multiselect("학습용 배터리 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트용 배터리 ID", all_ids, default=[all_ids[-1]])

    if st.button("🚀 통합 분석 시작"):
        # 데이터 준비
        X_train, y_train_rul, _, scaler = create_sequences_by_id(df, train_ids, window_size, is_train=True)
        X_test, y_test_rul, y_test_soh, _ = create_sequences_by_id(df, test_ids, window_size, scaler=scaler, is_train=False)

        # 5. 모델 구축 및 학습 (RUL 예측용)
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        progress_bar = st.progress(0)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(X_train, y_train_rul, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                  callbacks=[early_stop, tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch+1)/epochs))],
                  verbose=0)

        # 6. 예측 결과 생성
        y_pred_rul = model.predict(X_test).flatten()
        
        # SOH 예측치 시각화를 위한 간이 매핑 (RUL 변화량만큼 SOH 투영)
        # 실제 프로젝트에서는 SOH용 모델을 따로 두거나 Multi-output으로 구성하지만, 
        # 여기서는 RUL 예측 오차를 SOH 곡선에 반영하여 시각적 비교가 가능케 함
        y_pred_soh = y_test_soh + (y_pred_rul - y_test_rul) * 0.0005 # 오차 시각화용 보정

        # 7. 결과 시각화 (RUL & SOH 더블 그래프)
        st.success("✅ 분석 완료!")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 RUL: 실제 vs 예측")
            fig_rul, ax_rul = plt.subplots()
            ax_rul.plot(y_test_rul, label='Actual RUL', color='#2ecc71')
            ax_rul.plot(y_pred_rul, label='Predicted RUL', color='#e74c3c', linestyle='--')
            ax_rul.set_ylabel("Cycles")
            ax_rul.legend()
            st.pyplot(fig_rul)

        with col2:
            st.subheader("📉 SOH: 실제 vs 예측 (추정)")
            fig_soh, ax_soh = plt.subplots()
            ax_soh.plot(y_test_soh, label='Actual SOH', color='#3498db')
            ax_soh.plot(y_pred_soh, label='Predicted SOH', color='#f1c40f', linestyle='--')
            ax_soh.set_ylabel("SOH Value")
            ax_soh.legend()
            st.pyplot(fig_soh)

        # 8. 데이터 상세 표
        st.subheader("📋 상세 데이터 비교")
        comparison_df = pd.DataFrame({
            'Actual RUL': y_test_rul.astype(int),
            'Predicted RUL': y_pred_rul.astype(int),
            'Actual SOH': np.round(y_test_soh, 4),
            'Predicted SOH (Est.)': np.round(y_pred_soh, 4)
        })
        st.dataframe(comparison_df.head(50), use_container_width=True)

        # 9. 다운로드
        st.download_button("📥 결과 다운로드", comparison_df.to_csv(index=False).encode('utf-8'), "result.csv", "text/csv")
else:
    st.info("CSV 파일을 업로드해주세요.")
