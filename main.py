import streamlit as st
import pandas as pd


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random


import seaborn as sns
import matplotlib.pyplot as plt



# 사이드바 상태 관리 (세션 상태 활용)
if "predicted_price" not in st.session_state:
    st.session_state["predicted_price"] = None  # 초기값


# 간단한 데이터 및 모델 (더 복잡한 모델 및 데이터로 대체 가능)
def dummy_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(30, 4)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 더미 타임스텝 생성 함수
def create_dummy_timesteps(input_data_scaled, look_back=30):
    # 입력 데이터를 look_back만큼 복제하여 타임스텝을 만듦
    timesteps = np.repeat(input_data_scaled, look_back, axis=0)
    timesteps = timesteps.reshape(1, look_back, -1)  # (1, 30, 4)
    return timesteps

# 라인차트 예제
@st.cache_data
def make_line_chart(paramDf) :
    st.line_chart(
        paramDf,
        x = "quoter",
        y = ["sales", "margin"],
        color = ["#1764AB", "#4A98CA"],
        use_container_width = True,
    )

# 더미 데이터 만들기
@st.cache_data
def make_dummy_df() :
    return pd.DataFrame(
        {
            "quoter" : ["1Q", "2Q", "3Q", "4Q"],
            "sales" : random.sample(range(0, 100), 4),
            "margin" : random.sample(range(0, 100),4),
        }
    )



# 스케일러와 모델 (더 복잡한 데이터 준비 필요)
scaler = MinMaxScaler()
scaler.fit(np.random.rand(100, 5))  # 더미 스케일러 (BTC, S&P500, Gold, Copper, Oil)
model = dummy_model()


# Streamlit UI Left
with st.sidebar :
    st.title("비트코인 가격 예측 Side")
    st.write("날짜와 지수 데이터를 입력하여 비트코인 향후 가격을 예측합니다.")

    # 사용자 입력
    date_input = st.date_input("예측할 날짜:")
    sp500_input = st.number_input("S&P500 지수 입력:")
    gold_input = st.number_input("금 선물 지수 입력:")
    copper_input = st.number_input("구리 선물 지수 입력:")
    oil_input = st.number_input("원유 선물 지수 입력:")

    predict_button = st.button("예측하기")

    if predict_button :
        input_data = np.array([[sp500_input, gold_input, copper_input, oil_input]])
        input_data_scaled = scaler.transform(np.hstack((np.zeros((1, 1)), input_data)))[:, 1:]  # 더미 비트코인 0값 포함 후 제외
        input_data_reshaped = create_dummy_timesteps(input_data_scaled)

        # 더미 예측값 생성 (실제 모델로 교체 필요)
        prediction = model.predict(input_data_reshaped)
        st.session_state["predicted_price"] = scaler.inverse_transform([[prediction[0][0], 0, 0, 0, 0]])[0][0]



# 예시
df = make_dummy_df()

# Streamlit App - Right
st.title("비트코인 가격 예측")

# 입력 데이터 처리
st.header("학습 데이터", divider="rainbow")
st.write(df)
st.markdown("<br/><br/><br/><br/><br/>",  unsafe_allow_html=True)


# 예측 결과
st.subheader("예측 결과", divider="rainbow")
if st.session_state["predicted_price"] is not None:
    st.success(f"예측된 비트코인 가격: ${st.session_state['predicted_price']:.2f}")
else:
    st.info("사이드바에서 값을 입력하고 예측 버튼을 클릭하세요.")


st.markdown("<br/><br/><br/><br/><br/>",  unsafe_allow_html=True)

st.header("시각화 ", divider="rainbow")
make_line_chart(df)