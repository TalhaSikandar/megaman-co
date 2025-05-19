import numpy as np
from data_utils import load_data
from lstm_layer import LSTMLayer
from metadata import MODEL_PATH, PREPROCESSED_DATA_PATH, MODEL_CHECKPOINT_PATH
from train import plot_predictions, evaluate

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
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for LSTM model.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the trained model.")
    parser.add_argument("--data_path", type=str, default=PREPROCESSED_DATA_PATH, help="Path to the preprocessed data.")
    args = parser.parse_args()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_path)
    model = load_model(args.model_path)
    preds = predict(model, X_test)
    print("Predictions:", preds[:10])
    print("Ground Truth:", y_test[:10].flatten())
    plot_predictions(y_test, preds, title="Predictions vs Ground Truth - ONE", save_path="predictions_one.png")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()