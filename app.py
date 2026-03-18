import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH Research Lab", layout="wide")
st.title("🔋 Bi-LSTM 기반 배터리 SOH 정밀 예측 플랫폼")
st.markdown("본 모델은 **Bidirectional LSTM** 구조를 사용하여 배터리 열화 곡선을 추종합니다.")

# 2. 사이드바 설정 (논문 기반 파라미터)
st.sidebar.header("🔬 학습 설정 (Default)")
WINDOW_SIZE = st.sidebar.slider("Window Size", 1, 50, 6)
EPOCHS = st.sidebar.number_input("Epochs", 1, 500, 150)
BATCH_SIZE = st.sidebar.number_input("Batch Size", 1, 128, 40)

# 3. 데이터 준비 및 시퀀스 생성 함수
def prepare_battery_data(df, train_ids, test_ids, seq_length):
    features = ['voltage', 'temperature', 'capacity']
    target = 'soh'
    
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # 학습 데이터 기준으로 스케일러 피팅
    train_df_raw = df[df['battery_id'].isin(train_ids)]
    scaler_features.fit(train_df_raw[features])
    scaler_target.fit(train_df_raw[[target]])

    def create_sequences(target_ids):
        X_list, y_list, cycle_list = [], [], []
        for b_id in target_ids:
            battery_df = df[df['battery_id'] == b_id].sort_values('cycle')
            if len(battery_df) <= seq_length: continue
            
            f_scaled = scaler_features.transform(battery_df[features])
            t_scaled = scaler_target.transform(battery_df[[target]])
            
            for i in range(len(f_scaled) - seq_length):
                X_list.append(f_scaled[i:i+seq_length])
                y_list.append(t_scaled[i+seq_length])
                cycle_list.append(battery_df['cycle'].iloc[i+seq_length])
        return np.array(X_list).astype(np.float32), np.array(y_list).astype(np.float32), np.array(cycle_list), scaler_target

    X_train, y_train, _, _ = create_sequences(train_ids)
    X_test, y_test, test_cycles, sc_target = create_sequences(test_ids)
    
    return X_train, y_train, X_test, y_test, test_cycles, sc_target

# 4. 파일 업로드 섹션
uploaded_file = st.file_uploader("NASA 배터리 CSV 데이터를 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_ids = sorted(df['battery_id'].unique())
    
    st.sidebar.markdown("---")
    train_ids = st.sidebar.multiselect("학습용 배터리 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트용 배터리 ID", all_ids, default=[all_ids[-1]])

    if st.button("🚀 논문 설정으로 학습 및 분석 시작"):
        if not train_ids or not test_ids:
            st.error("⚠️ 학습용과 테스트용 배터리를 선택해주세요.")
        else:
            # 데이터 준비
            X_train, y_train, X_test, y_test, test_cycles, scaler_target = prepare_battery_data(
                df, train_ids, test_ids, WINDOW_SIZE
            )

            # 5. 모델 구축
            model = Sequential([
                Input(shape=(WINDOW_SIZE, X_train.shape[2])),
                Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
                Dropout(0.1),
                Bidirectional(LSTM(64, activation='tanh')),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            # 6. 학습 진행 (Streamlit 연동)
            st.subheader(f"🧠 학습 진행 중 (Total Epochs: {EPOCHS})")
            bar = st.progress(0)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.1,
                callbacks=[early_stop, tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: bar.progress((e+1)/EPOCHS))],
                shuffle=True,
                verbose=0
            )

            # 7. 예측 및 역스케일링
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_target.inverse_transform(y_pred_scaled).flatten()
            y_actual = scaler_target.inverse_transform(y_test).flatten()

            # 8. 결과 시각화 및 테이블 출력 (NameError 방지를 위해 변수명 명확히 선언)
            st.success("✅ 분석 완료!")
            
            # 그래프 영역과 테이블 영역을 나눕니다.
            col_left, col_right = st.columns([7, 3])
            
            with col_left:
                st.subheader(f"📈 SOH 추종 결과 (Battery ID: {test_ids})")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(test_cycles, y_actual, label='Actual SOH', color='#3498db', linewidth=2)
                ax.plot(test_cycles, y_pred, label='Predicted SOH', color='#e67e22', linestyle='--')
                ax.set_xlabel("Cycle")
                ax.set_ylabel("SOH")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col_right:
                st.subheader("📋 상세 수치")
                res_df = pd.DataFrame({
                    'Cycle': test_cycles.astype(int),
                    'Actual': np.round(y_actual, 4),
                    'Predicted': np.round(y_pred, 4)
                })
                st.dataframe(res_df.head(30), use_container_width=True)
                
                mae = mean_absolute_error(y_actual, y_pred)
                st.metric("평균 절대 오차 (MAE)", f"{mae:.5f}")

            # 9. 결과 다운로드
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 결과 CSV 다운로드", csv_data, "soh_results.csv", "text/csv")

else:
    st.info("💡 CSV 파일을 업로드하여 정밀 분석을 시작하세요.")
