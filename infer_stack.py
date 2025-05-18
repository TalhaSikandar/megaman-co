import numpy as np
from lstm_stack import StackedLSTMLayer
from metadata import MODEL_PATH, PREPROCESSED_DATA_DIR
from data_utils import load_data

def load_model(path):
    import pickle
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model

def infer(model, X):
    preds = []
    for i in range(len(X)):
        x_seq = X[i]
        y_pred = model.forward(x_seq, training=False)
        preds.append(y_pred.item())
    return np.array(preds)

def main():
    # Load data (adapt as needed for your inference scenario)
    _, _, _, _, X_test, y_test = load_data(PREPROCESSED_DATA_DIR)
    model = load_model(MODEL_PATH)
    preds = infer(model, X_test)
    print("Predictions:", preds)
    # Optionally compare with y_test
    if y_test is not None:
        y_true_flat = y_test.flatten()
        mae = np.mean(np.abs(preds - y_true_flat))
        print(f"MAE: {mae:.6f}")

if __name__ == "__main__":
    main()