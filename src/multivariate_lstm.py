import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def make_windows(X, y, window=30, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - window, step):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

def train_global_lstm(daily_df, features=['Temperature_C','Humidity_pct','Precipitation_mm','Wind_Speed_kmh'], window=30):
    df = daily_df.copy()
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df = df.sort_values(['Location','Date_Time'])
    loc_dummies = pd.get_dummies(df['Location'], prefix='loc')
    X_full = pd.concat([df[features].reset_index(drop=True), loc_dummies.reset_index(drop=True)], axis=1).astype(float)
    y_full = df['Temperature_C'].values
    mask = ~np.isnan(y_full)
    X_full = X_full[mask]; y_full = y_full[mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_windows, y_windows = make_windows(X_scaled, y_full, window=window)
    split = int(len(X_windows) * 0.9)
    X_train, X_test = X_windows[:split], X_windows[split:]
    y_train, y_test = y_windows[:split], y_windows[split:]
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=64, callbacks=[es])
    y_pred = model.predict(X_test).squeeze()
    return model, scaler, (X_test, y_test, y_pred)
