import streamlit as st
import pandas as pd
from src.preprocess import load_and_clean, resample_daily
from src.prophet_models import prophet_per_city

st.title("Weather Forecast ")
df = load_and_clean("data/weather_data.csv")
daily = resample_daily(df)
city = st.selectbox("Choose location", daily['Location'].unique())
if st.button("Run Prophet for city"):
    results = prophet_per_city(daily, test_days=30)
    if city in results:
        r = results[city]
        st.write("MAE:", r['metrics']['MAE'], "RMSE:", r['metrics']['RMSE'])
        model = r['model']; fcst = r['fcst_df']
        fig = model.plot(fcst)
        st.pyplot(fig)
    else:
        st.warning("Not enough data for this city")
