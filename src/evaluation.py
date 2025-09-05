import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def evaluate_series(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true,y_pred),
        "RMSE": rmse(y_true,y_pred),
        "MAPE": mape(y_true,y_pred)
    }
