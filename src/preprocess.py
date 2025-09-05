import pandas as pd
import numpy as np

def load_and_clean(path="data/weather_data.csv"):
    df = pd.read_csv(path)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
    df = df.dropna(subset=['Date_Time']).copy()
    numeric_cols = ['Temperature_C','Humidity_pct','Precipitation_mm','Wind_Speed_kmh']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(['Location','Date_Time']).reset_index(drop=True)
    df[numeric_cols] = df.groupby('Location')[numeric_cols].transform(lambda g: g.ffill().bfill())

    return df

def resample_daily(df):
    daily = (df.groupby('Location')
               .resample('D', on='Date_Time')
               .mean(numeric_only=True)
               .reset_index())
    return daily

if __name__ == "__main__":
    df = load_and_clean("data/weather_data.csv")
    daily = resample_daily(df)
    daily.to_csv("data/weather_daily.csv", index=False)
    print("Saved data/weather_daily.csv")
