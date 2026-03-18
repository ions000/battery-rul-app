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
st.set_page_config(page_title="Battery RUL Lab", layout="wide")
st.title("🔋 NASA 배터리 RUL 예측 플랫폼 (LSTM)")
st.markdown("""
이 앱은 여러 개의 배터리 데이터를 분리하여 학습하고, 특정 배터리의 남은 수명(RUL)을 예측합니다.
1. 데이터를 업로드하고 2. 학습/테스트용 배터리 ID를 선택한 뒤 3. 모델을 학습시키세요.
""")

# 2. 사이드바 설정
st.sidebar.header("⚙️ 모델 하이퍼파라미터")
window_size = st.sidebar.slider("Window Size (참조 사이클 수)", 5, 50, 10)
epochs = st.sidebar.number_input("Epochs (최대 학습 횟수)", 10, 200, 30)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# 3. 데이터 전처리 함수 (ID별 시퀀스 생성)
def create_sequences_by_id(df, selected_ids, window, scaler=None, is_train=True):
    X, y = [], []
    features = ['voltage', 'temperature', 'capacity', 'soh']
    
    # 학습 데이터일 경우에만 스케일러를 새로 피팅함
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

# 4. 파일 업로드 섹션
uploaded_file = st.file_uploader("NASA 배터리 데이터(CSV)를 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 데이터 미리보기")
    st.dataframe(df.head())

    # 배터리 ID 추출 및 선택
    all_ids = sorted(df['battery_id'].unique())
    
    st.sidebar.markdown("---")
    st.sidebar.header("📂 데이터 분할")
    train_ids = st.sidebar.multiselect("학습용 배터리 ID 선택", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트용 배터리 ID 선택 (결과 확인용)", all_ids, default=[all_ids[-1]])

    if st.button("🚀 모델 학습 및 예측 시작"):
        if not train_ids or not test_ids:
            st.error("⚠️ 학습용과 테스트용 배터리를 각각 하나 이상 선택해야 합니다.")
        else:
            with st.spinner('데이터 전처리 중...'):
                # 학습 데이터 생성
                X_train, y_train, scaler = create_sequences_by_id(df, train_ids, window_size, is_train=True)
                # 테스트 데이터 생성 (학습 때 사용한 스케일러 그대로 사용)
                X_test, y_test, _ = create_sequences_by_id(df, test_ids, window_size, scaler=scaler, is_train=False)

            st.write(f"✅ 학습 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")

            # 5. 모델 구축
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            # 6. 학습 진행 상황 표시
            st.subheader("🧠 모델 학습 중 (LSTM Training)")
            progress_bar = st.progress(0)
            status_text = st.empty()

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    percent = (epoch + 1) / epochs
                    progress_bar.progress(min(percent, 1.0))
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f}")

            early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop, StreamlitCallback()],
                verbose=0
            )

            # 7. 예측 및 평가
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.success("🎉 학습 및 예측 완료!")

            # 결과 지표 출력
            m1, m2 = st.columns(2)
            m1.metric("평균 절대 오차 (MAE)", f"{mae:.2f} Cycles")
            m2.metric("제곱근 평균 제곱 오차 (RMSE)", f"{rmse:.2f} Cycles")

            # 8. 시각화
            st.subheader(f"📈 예측 결과 시각화 (Test ID: {test_ids})")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_test, label='Actual RUL', color='#2ecc71', linewidth=2)
            ax.plot(y_pred, label='Predicted RUL', color='#e74c3c', linestyle='--')
            ax.set_xlabel('Time Steps (Cycles)')
            ax.set_ylabel('Remaining Useful Life (RUL)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # 학습 로그 보기
            with st.expander("📉 학습 손실 곡선 확인"):
                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(history.history['loss'], label='Train Loss')
                ax_loss.plot(history.history['val_loss'], label='Val Loss')
                ax_loss.legend()
                st.pyplot(fig_loss)
else:
    st.info("💡 CSV 파일을 업로드하면 배터리 ID별 학습 설정을 시작할 수 있습니다.")
