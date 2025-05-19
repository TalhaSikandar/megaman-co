from data_utils import load_data
from lstm_stack import StackedLSTMLayer
from config import EPOCHS, LEARNING_RATE
from metadata import MODEL_CHECKPOINT_DIR, MODEL_CHECKPOINT_PATH, MODEL_METADATA_PATH, MODEL_STACK_CHECKPOINT_PATH, MODEL_STACK_CHECKPOINT_DIR
from metadata import PREPROCESSED_DATA_PATH, MODEL_PATH, MODEL_STACK_PATH, MODEL_STACK_METADATA_PATH

def train(model, X_train, y_train, X_val, y_val, 
          epochs=10, start_epoch=0, 
          lr=0.001, 
          verbose=1, 
          checkpoint_path=None,
          patience=50):
    losses = []
    val_losses = []
    counter = 0
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            x_seq = X_train[i]
            y_true = y_train[i].reshape(-1, 1)
            y_pred = model.forward(x_seq, training=True)
            loss = model.compute_loss(y_true)
            epoch_loss += loss
            model.backward(y_true, learning_rate=lr)
        model.current_epoch = epoch 
        avg_loss = epoch_loss / len(X_train)
        losses.append(avg_loss)
        model.losses.append(avg_loss)

        # Validation loss
        val_loss = 0
        for i in range(len(X_val)):
            x_seq = X_val[i]
            y_true = y_val[i].reshape(-1, 1)
            y_pred = model.forward(x_seq, training=False)
            val_loss += model.compute_loss(y_true)
        avg_val_loss = val_loss / len(X_val)
        val_losses.append(avg_val_loss)
        model.val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

        # Save checkpoint if validation loss improves
        if checkpoint_path and avg_val_loss < model.best_val_loss:
            model.best_val_loss = avg_val_loss
            counter = 0
            save_model(model, checkpoint_path)
            if verbose:
                print(f"Checkpoint saved at epoch {epoch+1} with val loss {avg_val_loss:.6f}")
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print("Early stopping due to no improvement")
            break

    return losses, val_losses

def save_model(model, path, model_name=None):
    import pickle
    import os
    if model_name:
        path = path + "/" + model_name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path, model_name=None):
    import pickle
    if model_name:
        path = path + "/" + model_name
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model

def evaluate(model, X_test, y_test):
    import numpy as np
    test_loss = 0
    preds = []
    for i in range(len(X_test)):
        x_seq = X_test[i]
        y_true = y_test[i].reshape(-1, 1)
        y_pred = model.forward(x_seq, training=False)
        test_loss += model.compute_loss(y_true)
        preds.append(y_pred.item())
    avg_test_loss = test_loss / len(X_test)
    preds = np.array(preds)
    y_true_flat = y_test.flatten()
    mae = np.mean(np.abs(preds - y_true_flat))
    rmse = np.sqrt(np.mean((preds - y_true_flat) ** 2))
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    tolerance = 0.04
    accurate = np.abs(preds - y_true_flat) < (tolerance * np.abs(y_true_flat))
    accuracy = np.mean(accurate)
    print(f"Tolerance Accuracy : {accuracy:.2%}")

    return {
        "test_loss": avg_test_loss,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy
    }

def save_plots(model, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(model.losses, label='Train Loss')
    plt.plot(model.val_losses, label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + "/losses.png")
    plt.close()

import os
import json

def save_metadata(metadata, path, filename=None):
    if filename:
        path = path + "/" + filename

    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                existing_metadata = json.load(f)
                if not isinstance(existing_metadata, list):
                    existing_metadata = [existing_metadata]
            except Exception:
                existing_metadata = []
    else:
        existing_metadata = []
    existing_metadata.append(metadata)
    with open(path, "w") as f:
        json.dump(existing_metadata, f, indent=4)
    print(f"Metadata saved to {path}")
def print_data_values(x, y):
    print("X-shape :", x.shape, "\n", "X :", x[:10])
    print("Y-shape :", x.shape, "\n", "Y :", y[:10])

def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(PREPROCESSED_DATA_PATH)
    print("Loaded data")
    # print_data_values(X_train, y_train)
    input_size = X_train.shape[2]
    output_size = 1

    # Example stack config
    layer_configs = [
        {'hidden_size': 32, 'dropout': 0.1},
        {'hidden_size': 64, 'dropout': 0.2},
        {'hidden_size': 128, 'dropout': 0.3},
        # {'hidden_size': 120, 'dropout': 0.0},
    ]

    import argparse
    parser = argparse.ArgumentParser(description="Train Stacked LSTM model")
    parser.add_argument("--load_ckpt", type=str, default=None, help="Path to load model from")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()
    load_from_checkpoint = args.load_ckpt is not None

    epochs = 0
    lr = 0.001
    if load_from_checkpoint:
        model = load_model(MODEL_STACK_CHECKPOINT_PATH)
        epochs = model.epochs
        lr = model.learning_rate
        current_epoch = model.current_epoch
        print(f"Resuming training from epoch {current_epoch + 1}")
    else:
        model = StackedLSTMLayer(input_size, layer_configs, output_size)
        model.learning_rate = args.lr
        model.epochs = args.epochs
        model.current_epoch = 0 
        epochs = args.epochs
        current_epoch = 0
        lr = args.lr
        print("Training new model")
    train(model, X_train, y_train, X_val, y_val, epochs=epochs, start_epoch=current_epoch, lr=lr, checkpoint_path=MODEL_STACK_CHECKPOINT_PATH)
    save_model(model, MODEL_STACK_PATH)
    save_plots(model, MODEL_STACK_CHECKPOINT_DIR)
    test_metrics = evaluate(model, X_test, y_test)

    metadata = {
        "input_size": input_size,
        "layer_configs": layer_configs,
        "output_size": output_size,
        "epochs": epochs,
        "learning_rate": lr,
        "best_val_loss": min(model.val_losses) if hasattr(model, "val_losses") else None,
        "final_train_loss": model.losses[-1] if hasattr(model, "losses") else None,
        "test_metrics": test_metrics,
    }
    save_metadata(metadata, MODEL_STACK_METADATA_PATH)

if __name__ == "__main__":
    main()

# 