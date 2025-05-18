import numpy as np
from data_utils import load_data
from lstm_layer import LSTMLayer
from metadata import MODEL_PATH, PREPROCESSED_DATA_PATH, MODEL_CHECKPOINT_PATH

def load_model(path):
    import pickle
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, X):
    preds = []
    for i in range(len(X)):
        x_seq = X[i]
        y_pred = model.forward(x_seq)
        preds.append(y_pred.item())
    return np.array(preds)

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(PREPROCESSED_DATA_PATH)
    model = load_model(MODEL_CHECKPOINT_PATH)
    preds = predict(model, X_test)
    print("Predictions:", preds[:10])
    print("Ground Truth:", y_test[:10].flatten())

if __name__ == "__main__":
    main()