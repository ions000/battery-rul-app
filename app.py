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
from tensorflow.keras.optimizers import Adam

# 1. 페이지 설정
st.set_page_config(page_title="Battery RUL Fixer", layout="wide")
st.title("🔋 NASA 배터리 RUL 예측 (수렴 문제 해결 버전)")

# 2. 사이드바 설정 (디폴트 값)
st.sidebar.header("⚙️ 학습 설정")
window_size = st.sidebar.slider("Window Size", 1, 50, 6)
epochs = st.sidebar.number_input("Epochs", 1, 500, 150)
batch_size = st.sidebar.number_input("Batch Size", 1, 128, 40)

# 3. 데이터 전처리 함수 (타겟 스케일링 추가)
def prepare_data(df, train_ids, test_ids, window):
    features = ['voltage', 'temperature', 'capacity', 'soh']
    target = 'rul'
    
    # 특징(X) 스케일러와 타겟(y) 스케일러 분리
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 학습 데이터 기준으로 피팅
    train_df = df[df['battery_id'].isin(train_ids)].sort_values(['battery_id', 'cycle'])
    scaler_X.fit(train_df[features])
    scaler_y.fit(train_df[[target]])
    
    def get_sequences(target_ids):
        X_list, y_list, soh_list = [], [], []
        for b_id in target_ids:
            temp_df = df[df['battery_id'] == b_id].sort_values('cycle')
            if len(temp_df) <= window: continue
            
            x_scaled = scaler_X.transform(temp_df[features])
            y_scaled = scaler_y.transform(temp_df[[target]])
            soh_raw = temp_df['soh'].values
            
            for i in range(len(temp_df) - window):
                X_list.append(x_scaled[i:i+window])
                y_list.append(y_scaled[i+window])
                soh_list.append(soh_raw[i+window])
        return np.array(X_list), np.array(y_list), np.array(soh_list)

    X_train, y_train, _ = get_sequences(train_ids)
    X_test, y_test, soh_test = get_sequences(test_ids)
    
    return X_train, y_train, X_test, y_test, soh_test, scaler_y

# 4. 파일 업로드
uploaded_file = st.file_uploader("NASA 배터리 CSV 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_ids = sorted(df['battery_id'].unique())
    train_ids = st.sidebar.multiselect("학습 ID", all_ids, default=all_ids[:-1])
    test_ids = st.sidebar.multiselect("테스트 ID", all_ids, default=[all_ids[-1]])

    if st.button("🚀 예측 시작"):
        # 데이터 준비
        X_train, y_train, X_test, y_test, soh_test, scaler_y = prepare_data(df, train_ids, test_ids, window_size)

        # 5. 모델 구성 (최적화)
        model = Sequential([
            Input(shape=(window_size, X_train.shape[2])),
            Bidirectional(LSTM(32, return_sequences=False)), # 윈도우가 작으므로 구조 단순화
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='linear') # 타겟이 0~1 사이이므로 학습이 훨씬 쉬워짐
        ])
        
        # 학습률(LR)을 조금 낮추어 안정적 수렴 유도
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # 6. 학습
        status = st.empty()
        bar = st.progress(0)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            shuffle=True, # 데이터 섞기 추가
            callbacks=[early_stop, tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: bar.progress((e+1)/epochs))],
            verbose=0
        )

        # 7. 예측 및 역스케일링 (중요!)
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_actual = scaler_y.inverse_transform(y_test).flatten()

        # 8. 시각화
        st.success("✅ 학습 완료! RUL 값이 변하는지 확인하세요.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 RUL 예측 결과")
            fig, ax = plt.subplots()
            ax.plot(y_actual, label='Actual', color='green')
            ax.plot(y_pred, label='Predicted', color='red', linestyle='--')
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.subheader("📋 예측 데이터 표")
            res_df = pd.DataFrame({'Actual': y_actual.astype(int), 'Predicted': y_pred.astype(int)})
            st.dataframe(res_df.head(50), use_container_width=True)
            
        st.metric("평균 절대 오차 (MAE)", f"{mean_absolute_error(y_actual, y_pred):.2f} Cycles")
