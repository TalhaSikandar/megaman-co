---

# 📈 LSTM-Based Stock Price Predictor (Built From Scratch in Python)

This academic project implements a custom **Long Short-Term Memory (LSTM)** network from scratch in pure Python for **next-day stock price prediction**. Unlike most deep learning projects that rely on libraries like PyTorch or TensorFlow, this project builds all LSTM components manually — including weight matrices, gate logic, and backpropagation through time (BPTT). The goal was to gain a deep understanding of LSTM internals while solving a real-world time series regression problem.

---

## 🚀 Project Highlights

* 📊 **Next-day stock price regression** using historical financial data.
* 🧠 **LSTM architecture manually implemented** (no PyTorch, TensorFlow, or Keras).
* 🔁 **End-to-end training pipeline** with Adam optimizer, loss tracking, early stopping, and checkpointing.
* 📉 **Evaluation metrics** include MAE, RMSE, and accuracy under tolerance thresholds.
* 🧪 **Custom inference pipeline** to make predictions using trained models.
* 📂 **Modular code structure** separating data handling, model logic, training, and evaluation.

---

## 📦 Project Structure & Flow

### 1. **Data Loading**

* The script uses `load_data()` from `data_utils.py` to load preprocessed train/validation/test datasets.
* Input shape: `(num_samples, lookback_window, 6)`, where each sample has 6 financial features over a lookback window of days.

### 2. **Model Architecture**

* `LSTMLayer` implements:

  * A manually constructed LSTM cell with:

    * Input gate, forget gate, output gate, and candidate computation.
    * Cell and hidden state updates for each timestep.
  * A final **fully connected linear layer** to map the LSTM output to a single scalar (next day's price).

### 3. **Training Pipeline**

* Trains over user-defined number of epochs.
* Tracks training and validation loss (MSE).
* Implements:

  * **Adam optimizer**
  * **Early stopping**
  * **Checkpoint saving**
  * Hyperparameter logging (JSON)

### 4. **Evaluation**

* Evaluates the model on the test set using:

  * **Mean Absolute Error (MAE)**
  * **Root Mean Square Error (RMSE)**
  * **Tolerance-based accuracy** (within an acceptable error range)
* Metrics are printed and saved for analysis.

### 5. **Inference**

* `infer.py` loads the trained weights and performs predictions on test data.
* Supports inference consistency by matching preprocessing steps and architecture.

---

## 📌 Things to Ensure

> While the core logic is solid, some practical considerations are key to improving reliability:

* ✅ **Data normalization** is critical for LSTM performance.
* ✅ Ensure **training and inference pipelines share the same preprocessing steps**.
* 🔁 The model is saved/loaded via `pickle`, so architecture consistency is essential.
* ⚠️ Market behavior is **inherently noisy**, so real-world accuracy has limits.
* 🧠 Feature selection and engineering can make a significant difference.

---

## ✅ Will It Work?

**Yes — this LSTM-based model is technically sound and capable of learning temporal dependencies from historical data.**
It functions as a full-stack prototype for stock price prediction and serves as a hands-on deep learning systems project — giving full control over every parameter, gradient, and computation.

---

## 📄 Conclusion

This project demonstrates a complete and explainable deep learning pipeline implemented from scratch — ideal for educational use, low-level exploration, and experimental research. It trades speed and convenience for **transparency, full control, and deep understanding** of how LSTMs operate internally.

---
