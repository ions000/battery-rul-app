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
st.set_page_config(page_title="Battery SOH Zip Manager", layout="wide")
st.title("🔋 배터리 SOH 예측 (ZIP 모델 관리 버전)")

# 2. 모델 로드 함수 (ZIP 압축 해제 포함)
def load_model_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        # 모델 파일 추출 (.keras)
        model_name = [f for f in z.namelist() if f.endswith('.keras')][0]
        z.extract(model_name)
        model = load_model(model_name)
        
        # 스케일러 파일 추출 (.pkl)
        scaler_name = [f for f in z.namelist() if f.endswith('.pkl')][0]
        with z.open(scaler_name) as f:
            sc_X, sc_y = pickle.load(f)
            
    return model, sc_X, sc_y

# 3. 사이드바 - ZIP 파일 업로드
st.sidebar.header("📁 모델 불러오기")
uploaded_zip = st.sidebar.file_uploader("학습된 ZIP 파일을 올리세요", type="zip")

# 4. 데이터 전처리 함수 (기존과 동일)
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

# 5. 파일 업로드 및 메인 로직
uploaded_csv = st.file_uploader("NASA 배터리 CSV 업로드", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    all_ids = sorted(df['battery_id'].unique())
    test_ids = st.sidebar.multiselect("테스트 ID", all_ids, default=[all_ids[-1]])

    # 상황 A: ZIP 파일로 즉시 예측
    if uploaded_zip:
        if st.button("🔌 ZIP 모델로 즉시 예측"):
            model, sc_X, sc_y = load_model_from_zip(uploaded_zip)
            _, _, X_test, y_test, test_cycles, _, _ = prepare_data(df, [], test_ids, 6, sc_X, sc_y)
            y_pred = sc_y.inverse_transform(model.predict(X_test)).flatten()
            y_actual = sc_y.inverse_transform(y_test).flatten()
            st.session_state['res'] = (y_actual, y_pred, test_cycles)

    # 상황 B: 신규 학습 및 ZIP 저장
    if st.button("🚀 신규 학습 및 ZIP 저장"):
        train_ids = [i for i in all_ids if i not in test_ids]
        X_tr, y_tr, X_te, y_te, cycles, sc_X, sc_y = prepare_data(df, train_ids, test_ids, 6)
        
        model = Sequential([
            Input(shape=(6, X_tr.shape[2])),
            Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
            Bidirectional(LSTM(64, activation='tanh')),
            Dense(32, activation='relu'), Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_tr, y_tr, epochs=150, batch_size=40, verbose=0)
        
        y_pred = sc_y.inverse_transform(model.predict(X_te)).flatten()
        y_actual = sc_y.inverse_transform(y_te).flatten()
        st.session_state['res'] = (y_actual, y_pred, cycles)

        # ZIP 파일 생성 로직
        model.save("model.keras")
        with open("scalers.pkl", "wb") as f: pickle.dump((sc_X, sc_y), f)
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            z.write("model.keras")
            z.write("scalers.pkl")
        
        st.download_button("📥 통합 모델(ZIP) 다운로드", zip_buffer.getvalue(), "battery_package.zip")

    # 결과 출력
    if 'res' in st.session_state:
        y_a, y_p, cyc = st.session_state['res']
        st.line_chart(pd.DataFrame({'Actual': y_a, 'Predicted': y_p}, index=cyc))
        st.metric("MAE", f"{mean_absolute_error(y_a, y_p):.5f}")
