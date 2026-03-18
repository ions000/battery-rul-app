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
from tensorflow.keras.optimizers import Adam

# 1. 페이지 설정
st.set_page_config(page_title="Battery SOH Analyzer", layout="wide")
st.title("🔋 NASA 배터리 SOH(건강 상태) 예측 플랫폼")
st.markdown("본 도구는 LSTM을 활용하여 배터리의 SOH 퇴화를 예측하고 실제 데이터와 비교합니다.")

# 2. 사이드바 설정 (사용자 요청 디폴트 값)
st.sidebar.header("⚙️ 모델 학습 설정")
window_size = st.sidebar.slider("Window Size (과거 참조 사이클)", 1, 50, 6)
epochs = st.sidebar.number_input("Epochs (최대 학습 횟수)", 1, 500, 150)
batch_size = st.sidebar.number_input("Batch Size (배치 크기)", 1, 128, 40)

# 3. 데이터 전처리 함수 (SOH 타겟 설정)
def prepare_soh_data(df, train_ids, test_ids, window):
    features = ['voltage', 'temperature', 'capacity'] # 입력 특성
    target = 'soh' # 예측 목표
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 학습 데이터 기준으로 스케일러 학습
    train_df = df[df['battery_id'].isin(train_ids)].sort_values(['battery_id', 'cycle'])
    scaler_X.fit(train_df[features])
    scaler_y.fit(train_df[[target]])
    
    def get_sequences(target_ids):
        X_list, y_list = [], []
        for b_id in target_ids:
            temp_df = df[df['battery_id'] == b_id].sort_values('cycle')
            if len(temp_df) <= window: continue
            
            x_scaled = scaler_X.transform(temp_df[features])
            y_scaled = scaler_y.transform(temp_df[[target]])
            
            for i in range(len(temp_df) - window):
                X_list.append(x_scaled[i:i+window])
                y_list.append(y_scaled[i+window])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = get_sequences(train_ids)
    X_test, y_test = get_sequences(test_ids)
    
    return X_train, y_train, X_test, y_test, scaler_y

# 4. 파일 업로드 섹션
uploaded_file = st.file_uploader("NASA 배터리 CSV 데이터를 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_ids = sorted(df['battery_id'].unique())
    
    st.sidebar.markdown("---")
    train_ids = st.sidebar.multiselect("학습용 배터리 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트용 배터리 ID", all_ids, default=[all_ids[-1]])

    if st.button("🚀 SOH 예측 시작"):
        if not train_ids or not test_ids:
            st.error("⚠️ 학습용과 테스트용 배터리를 선택해주세요.")
        else:
            # 데이터 준비
            X_train, y_train, X_test, y_test, scaler_y = prepare_soh_data(df, train_ids, test_ids, window_size)

            # 5. 모델 구축
            model = Sequential([
                Input(shape=(window_size, X_train.shape[2])),
                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            # 6. 학습 진행 상황
            st.subheader("🧠 모델 학습 진행 중...")
            bar = st.progress(0)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      validation_split=0.1, shuffle=True,
                      callbacks=[early_stop, tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: bar.progress((e+1)/epochs))],
                      verbose=0)

            # 7. 예측 및 결과 복원
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            y_actual = scaler_y.inverse_transform(y_test).flatten()

            # 8. 결과 리포트 (그래프와 표)
            st.success("✅ SOH 예측 분석 완료!")
            
            col_graph, col_table = st.columns([7, 3])
            
            with col_graph:
                st.subheader(f"📈 SOH 예측 결과 (Test ID: {test_ids})")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(y_actual, label='Actual SOH', color='#3498db', linewidth=2)
                ax.plot(y_pred, label='Predicted SOH', color='#f39c12', linestyle='--')
                ax.set_xlabel("Cycles")
                ax.set_ylabel("SOH (State of Health)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col_table:
                st.subheader("📋 수치 비교")
                res_df = pd.DataFrame({
                    'Actual SOH': np.round(y_actual, 4),
                    'Predicted SOH': np.round(y_pred, 4)
                })
                res_df['Error'] = np.abs(res_df['Actual SOH'] - res_df['Predicted SOH'])
                st.dataframe(res_df.head(20), use_container_width=True)
                
                mae = mean_absolute_error(y_actual, y_pred)
                st.metric("평균 절대 오차 (MAE)", f"{mae:.4f}")

            # 9. 결과 다운로드
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 SOH 예측 결과 다운로드", csv_data, "soh_prediction.csv", "text/csv")
else:
    st.info("💡 CSV 파일을 업로드하여 SOH 예측을 시작하세요.")
