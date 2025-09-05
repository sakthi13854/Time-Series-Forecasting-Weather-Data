import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.evaluation import evaluate_series
import warnings
warnings.filterwarnings("ignore")

def train_arima_per_city(daily_df, target='Temperature_C', test_days=30, arima_order=(5,1,0)):
    results = {}
    for city in daily_df['Location'].unique():
        city_df = daily_df[daily_df['Location']==city].set_index('Date_Time').sort_index()
        city_df.index = pd.to_datetime(city_df.index)
        city_df = city_df.asfreq('D')
        series = city_df[target].dropna()
        if len(series) < (test_days + 50):
            continue
        train = series.iloc[:-test_days]
        test = series.iloc[-test_days:]
        model = ARIMA(train, order=arima_order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=test_days)
        metrics = evaluate_series(test.values, forecast.values)
        results[city] = {"metrics": metrics, "forecast": forecast, "test": test}
    return results
