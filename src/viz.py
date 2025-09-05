import matplotlib.pyplot as plt

def plot_forecast(test_index, y_true, y_pred, title="Forecast vs Actual"):
    plt.figure(figsize=(12,5))
    plt.plot(test_index, y_true, label='Actual')
    plt.plot(test_index, y_pred, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.show()
