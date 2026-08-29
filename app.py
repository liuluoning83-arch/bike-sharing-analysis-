"""Interactive rental-demand prediction demo."""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from src.model import FEATURES


MODEL_PATH = Path("artifacts/bike_rental_model.joblib")
WEATHER_OPTIONS = {
    "晴朗 / 少云": 1,
    "薄雾 / 多云": 2,
    "小雨或小雪": 3,
    "强降雨或强降雪": 4,
}


def season_from_month(month: int) -> int:
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    if month in (9, 10, 11):
        return 3
    return 4


def build_feature_row(
    selected_date: date,
    weather: int,
    holiday: bool,
    temperature_c: float,
    feels_like_c: float,
    humidity_percent: int,
    wind_speed_kmh: float,
) -> pd.DataFrame:
    # The original dataset stores weather values as normalized measurements.
    weekday = (selected_date.weekday() + 1) % 7  # Sunday is 0 in the dataset.
    is_working_day = int(weekday not in (0, 6) and not holiday)
    row = {
        "season": season_from_month(selected_date.month),
        # The historical dataset has 0 for 2011 and 1 for 2012. Later dates use
        # the 2012 demand level, so this is a demonstration rather than a live forecast.
        "yr": int(selected_date.year >= 2012),
        "mnth": selected_date.month,
        "holiday": int(holiday),
        "weekday": weekday,
        "workingday": is_working_day,
        "weathersit": weather,
        "temp": (temperature_c + 8) / 47,
        "atemp": (feels_like_c + 16) / 66,
        "hum": humidity_percent / 100,
        "windspeed": wind_speed_kmh / 67,
    }
    return pd.DataFrame([row], columns=FEATURES)


st.set_page_config(page_title="共享单车需求预测", page_icon="🚲")
st.title("🚲 共享单车租赁需求预测")
st.caption("基于 2011—2012 年日级数据训练的随机森林演示模型")

if not MODEL_PATH.exists():
    st.error("尚未找到模型文件。请先在终端运行：python -m src.train_model")
    st.stop()

with st.sidebar:
    st.header("输入条件")
    date_text = st.text_input(
        "日期（YYYY-MM-DD）",
        value="2012-09-15",
        help="例如：2012-09-15。可以直接修改文本，不需要使用日历。",
    ).strip()
    try:
        selected_date = date.fromisoformat(date_text)
    except ValueError:
        st.error("日期格式不正确，请使用 YYYY-MM-DD，例如 2012-09-15。")
        st.stop()
    weather_name = st.selectbox("天气", list(WEATHER_OPTIONS))
    holiday = st.checkbox("是否法定节假日")
    temperature_c = st.slider("实际温度（°C）", -8, 39, 22)
    feels_like_c = st.slider("体感温度（°C）", -16, 50, 24)
    humidity_percent = st.slider("湿度（%）", 0, 100, 60)
    wind_speed_kmh = st.slider("风速（km/h）", 0, 67, 15)

model = load(MODEL_PATH)
input_data = build_feature_row(
    selected_date,
    WEATHER_OPTIONS[weather_name],
    holiday,
    temperature_c,
    feels_like_c,
    humidity_percent,
    wind_speed_kmh,
)
prediction = max(0, round(float(model.predict(input_data)[0])))

st.metric("预测日租赁量", f"{prediction:,} 次")

st.subheader("本次预测使用的特征")
st.dataframe(input_data, use_container_width=True, hide_index=True)

st.info(
    "这是学习项目的演示预测。模型未包含实时天气、节假日活动、区域车辆供给等信息；"
    "对 2012 年后的日期会沿用 2012 年的需求水平，因此不能直接用于真实运营决策。"
)
