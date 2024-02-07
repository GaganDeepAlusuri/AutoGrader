
# Long Short-Term Memory Network for Time Series Prediction

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Generate synthetic time series data
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # Wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # Wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # Noise
    return series[..., np.newaxis].astype(np.float32)

# Prepare the data
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=[None, 1]),
    LSTM(50),
    Dense(1)
])

# Compile and train the model
model.compile(optimizer=Adam(lr=0.01), loss='mse')
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Evaluate the model
mse_test = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse_test}")

# Plot some predictions
def plot_series(series, y=None, y_pred=None, x_label='$t$', y_label='$x(t)$'):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0],
                model.predict(X_valid[col:col+1])[0, 0],
                x_label=("$t$" if col==0 else None),
                y_label=("$x(t)$" if col==0 else None))
plt.show()
