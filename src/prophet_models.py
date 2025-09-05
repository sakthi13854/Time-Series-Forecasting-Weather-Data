from prophet import Prophet
import pandas as pd
from src.evaluation import evaluate_series

def prophet_per_city(daily_df, target='Temperature_C', test_days=30):
    results = {}
    for city in daily_df['Location'].unique():
        city_df = daily_df[daily_df['Location']==city][['Date_Time', target]].dropna().rename(columns={'Date_Time':'ds', target:'y'})
        if len(city_df) < (test_days + 50):
            continue
        train = city_df.iloc[:-test_days]
        test = city_df.iloc[-test_days:]
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(train)
        future = m.make_future_dataframe(periods=test_days, freq='D')
        fcst = m.predict(future)
        y_pred = fcst['yhat'].values[-test_days:]
        metrics = evaluate_series(test['y'].values, y_pred)
        results[city] = {"metrics": metrics, "forecast": y_pred, "test": test['y'].values, "model": m, "fcst_df": fcst}
    return results
