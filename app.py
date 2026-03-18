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
st.title("🔋 NASA 배터리 RUL 예측 및 데이터 분석 플랫폼")
st.markdown("""
이 대시보드는 LSTM 모델을 사용하여 배터리의 잔여 수명(RUL)을 예측합니다. 
학습 데이터와 테스트 데이터를 분리하여 모델의 일반화 성능을 확인하고, 상세 예측 수치를 표로 비교해 보세요.
""")

# 2. 사이드바 설정 (요청하신 하이퍼파라미터 디폴트 값 적용)
st.sidebar.header("⚙️ 모델 학습 설정")
window_size = st.sidebar.slider("Window Size (과거 참조 사이클)", 1, 50, 6) # Default: 6
epochs = st.sidebar.number_input("Epochs (최대 학습 횟수)", 1, 500, 150) # Default: 150
batch_size = st.sidebar.number_input("Batch Size (배치 크기)", 1, 128, 40) # Default: 40

# 3. 데이터 전처리 함수
def create_sequences_by_id(df, selected_ids, window, scaler=None, is_train=True):
    X, y = [], []
    features = ['voltage', 'temperature', 'capacity', 'soh']
    
    if is_train:
        scaler = MinMaxScaler()
        scaler.fit(df[df['battery_id'].isin(selected_ids)][features])
    
    for b_id in selected_ids:
        temp_df = df[df['battery_id'] == b_id].sort_values('cycle')
        # 데이터가 윈도우 사이즈보다 작을 경우 스킵
        if len(temp_df) <= window:
            continue
            
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
    all_ids = sorted(df['battery_id'].unique())
    
    st.sidebar.markdown("---")
    st.sidebar.header("📂 데이터 분할 설정")
    train_ids = st.sidebar.multiselect("학습에 사용할 배터리 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트할 배터리 ID", all_ids, default=[all_ids[-1]])

    if st.button("🚀 모델 학습 및 상세 분석 시작"):
        if not train_ids or not test_ids:
            st.error("⚠️ 학습용과 테스트용 배터리를 각각 선택해야 합니다.")
        else:
            with st.spinner('데이터를 정렬하고 시퀀스를 생성하는 중...'):
                X_train, y_train, scaler = create_sequences_by_id(df, train_ids, window_size, is_train=True)
                X_test, y_test, _ = create_sequences_by_id(df, test_ids, window_size, scaler=scaler, is_train=False)

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

            # 6. 학습 진행 표시
            st.subheader("🧠 모델 학습 상태")
            progress_bar = st.progress(0)
            status_text = st.empty()

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f}")

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop, StreamlitCallback()],
                verbose=0
            )

            # 7. 예측 및 결과 데이터 정리
            y_pred = model.predict(X_test)
            
            # 실제값과 예측값을 정수로 변환하여 비교표 생성
            comparison_df = pd.DataFrame({
                'Actual RUL': y_test.flatten().astype(int),
                'Predicted RUL': y_pred.flatten().astype(int)
            })
            comparison_df['Error'] = comparison_df['Actual RUL'] - comparison_df['Predicted RUL']
            comparison_df['Abs Error'] = comparison_df['Error'].abs()

            st.success("✅ 분석 완료!")

            # 8. 시각화 및 테이블 출력 (7:3 레이아웃)
            st.subheader(f"📊 RUL 예측 상세 리포트 (Test ID: {test_ids})")
            
            col_graph, col_table = st.columns([7, 3])

            with col_graph:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(y_test, label='Actual RUL', color='#2ecc71', linewidth=2)
                ax.plot(y_pred, label='Predicted RUL', color='#e74c3c', linestyle='--')
                ax.set_title(f"Battery RUL Prediction (Window: {window_size}, Epochs: {epochs})")
                ax.set_xlabel("Time Steps (Cycles)")
                ax.set_ylabel("Remaining Cycles")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col_table:
                st.write("📋 수치 비교 (상위 20개 행)")
                st.dataframe(comparison_df.head(20), use_container_width=True)
                
                mae = comparison_df['Abs Error'].mean()
                st.metric("평균 절대 오차 (MAE)", f"{mae:.2f} Cycles")

            # 9. SOH 퇴화 그래프 추가 시각화
            with st.expander("📉 SOH(건강 상태) 퇴화 곡선 보기"):
                fig_soh, ax_soh = plt.subplots(figsize=(10, 4))
                for b_id in test_ids:
                    soh_data = df[df['battery_id'] == b_id].sort_values('cycle')['soh']
                    ax_soh.plot(soh_data.values, label=f'Battery {b_id} SOH')
                ax_soh.axhline(y=0.8, color='red', linestyle='--', label='Limit (0.8)')
                ax_soh.set_ylabel("SOH Value")
                ax_soh.legend()
                st.pyplot(fig_soh)

            # 10. 결과 다운로드
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 예측 결과 CSV 다운로드",
                data=csv,
                file_name=f'rul_prediction_results_id_{test_ids}.csv',
                mime='text/csv',
            )
else:
    st.info("💡 사이드바에서 하이퍼파라미터를 확인하고, NASA 배터리 CSV 데이터를 업로드해 주세요.")
