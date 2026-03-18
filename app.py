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
st.set_page_config(page_title="Battery RUL Predictor", layout="wide")
st.title("🔋 LSTM 기반 배터리 잔여 수명(RUL) 예측 연구소")
st.write("NASA 데이터를 활용하여 배터리의 SOH와 RUL을 예측하는 딥러닝 모델을 직접 학습시켜보세요.")

# 2. 사이드바 - 하이퍼파라미터 설정
st.sidebar.header("🛠️ 모델 설정")
window_size = st.sidebar.slider("Window Size (과거 참조 사이클)", 5, 30, 10)
epochs = st.sidebar.number_input("Epochs (학습 횟수)", 10, 100, 20)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# 3. 데이터 로드 및 전처리 함수
def prepare_data(df, window):
    # 필요한 컬럼만 추출 (사용자 데이터 구조에 맞춤)
    features = ['voltage', 'temperature', 'capacity', 'soh']
    target = 'rul'
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    scaled_target = df[target].values # RUL은 보통 스케일링하지 않거나 별도로 함
    
    X, y = [], []
    for i in range(len(df) - window):
        X.append(scaled_features[i:i+window])
        y.append(scaled_target[i+window])
    
    return np.array(X), np.array(y), scaler

# 4. 파일 업로드
uploaded_file = st.file_uploader("NASA 배터리 CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 업로드된 데이터 샘플")
    st.dataframe(df.head())

    if st.button("🚀 LSTM 모델 학습 및 예측 시작"):
        # 데이터 준비
        X, y, scaler = prepare_data(df, window_size)
        
        # Train/Test 분할 (8:2)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 5. 모델 구축 (요청하신 Bidirectional 추가)
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # 6. 학습 진행 표시
        st.subheader("🧠 모델 학습 중...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 학습 과정을 모니터링하기 위한 간단한 콜백
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                percent = (epoch + 1) / epochs
                progress_bar.progress(percent)
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f}")

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop, StreamlitCallback()],
            verbose=0
        )

        # 7. 예측 및 결과 확인
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.success("✅ 학습 완료!")
        
        # 결과 메트릭 출력
        col1, col2 = st.columns(2)
        col1.metric("MAE (평균 절대 오차)", f"{mae:.2f} Cycles")
        col2.metric("RMSE (제곱근 평균 제곱 오차)", f"{rmse:.2f} Cycles")

        # 8. 결과 시각화
        st.subheader("📈 RUL 예측 결과 그래프")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test, label='Actual RUL', color='#1f77b4', linewidth=2)
        ax.plot(y_pred, label='Predicted RUL', color='#ff7f0e', linestyle='--')
        ax.set_xlabel('Test Samples (Cycles)')
        ax.set_ylabel('Remaining Useful Life')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 9. 학습 손실 곡선
        with st.expander("학습 손실(Loss) 곡선 보기"):
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history.history['loss'], label='Train Loss')
            ax_loss.plot(history.history['val_loss'], label='Val Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)

else:
    st.info("왼쪽 사이드바에서 설정을 확인하고, CSV 데이터를 업로드하여 시작하세요.")