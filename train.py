from data_utils import load_data
from lstm_layer import LSTMLayer
from config import EPOCHS, LEARNING_RATE
from metadata import MODEL_CHECKPOINT_DIR, MODEL_CHECKPOINT_PATH, MODEL_METADATA_PATH
from metadata import PREPROCESSED_DATA_PATH, MODEL_PATH

### Suggestions

# - Add docstrings to functions for clarity.

def train(model, X_train, y_train, X_val, y_val, 
          epochs=10, start_epoch=0, 
          lr=0.001, 
          verbose=1, 
          checkpoint_path=None,
          patience=20):
    losses = []
    val_losses = []
    counter = 0
    for epoch in range(start_epoch, epochs):
        y_preds = []
        y_trues = []
        epoch_loss = 0
        for i in range(len(X_train)):
            x_seq = X_train[i]
            y_true = y_train[i].reshape(-1, 1)
            y_trues.append(y_true)
            y_pred = model.forward(x_seq)
            y_preds.append(y_pred)
            # print("y_pred", y_pred)
            # print("y_true", y_true)
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
            y_pred = model.forward(x_seq)  # Just forward pass
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
            save_model(model, MODEL_CHECKPOINT_PATH)
            if verbose:
                print(f"Checkpoint saved at epoch {epoch+1} with val loss {avg_val_loss:.6f}")
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print("Early stopping due to no improvement")
            break

    return losses, val_losses, y_trues, y_preds

def save_model(model, path, model_name=None):
    import pickle
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
        y_pred = model.forward(x_seq)
        test_loss += model.compute_loss(y_true)
        preds.append(y_pred.item())
    avg_test_loss = test_loss / len(X_test)
    preds = np.array(preds)
    y_true_flat = y_test.flatten()
    mae = np.mean(np.abs(preds - y_true_flat))
    rmse = np.sqrt(np.mean((preds - y_true_flat) ** 2))
    # print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    tolerance = 0.09
    accurate = np.abs(preds - y_true_flat) < (tolerance * np.abs(y_true_flat))
    accuracy = np.mean(accurate)
    print(f"Tolerance Accuracy: {accuracy:.2%}")

    return {
        "test_loss": avg_test_loss,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy
    }

# Draw and save all kinds of necessary plots
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

    # Load the json list and append to it
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

def plot_predictions(y_true, y_pred, title="Stock Price Prediction", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label="True Price", color="blue")
    plt.plot(y_pred, label="Predicted Price", color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(PREPROCESSED_DATA_PATH)
    input_size = X_train.shape[2]
    hidden_size = 64
    output_size = 1
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model")
    # Boolean flag to load model from checkpoint
    parser.add_argument("--load_ckpt", type=str, default=None, help="Path to load model from")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")

    args = parser.parse_args()
    load_from_checkpoint = args.load_ckpt is not None

    epochs = 0
    # lr = 0.001
    if load_from_checkpoint:
        model = load_model(MODEL_CHECKPOINT_PATH)
        epochs = model.epochs
        lr = model.learning_rate
        current_epoch = model.current_epoch
        print(f"Resuming training from epoch {current_epoch + 1}")

    else:
        # Initialize model
        model = LSTMLayer(input_size, hidden_size, output_size)
        model.learning_rate = args.lr
        model.epochs = args.epochs
        model.current_epoch = 0 
        epochs = args.epochs
        current_epoch = 0
        lr = args.lr
    _, _, y_trues, y_preds = train(model, X_train, y_train, X_val, y_val, epochs=epochs, start_epoch=current_epoch, lr=lr, checkpoint_path=MODEL_CHECKPOINT_DIR)
    save_model(model, MODEL_PATH)
    save_plots(model, MODEL_CHECKPOINT_DIR)
    # Plot predictions
    plot_predictions(y_true=y_trues, y_pred=y_preds, title="Stock Price Prediction", save_path=MODEL_CHECKPOINT_DIR + "/predictions.png")   
    test_metrics = evaluate(model, X_test, y_test)

    # Save metadata
    metadata = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "epochs": epochs,
        "learning_rate": lr,
        "best_val_loss": min(model.val_losses) if hasattr(model, "val_losses") else None,
        "final_train_loss": model.losses[-1] if hasattr(model, "losses") else None,
        "test_metrics": test_metrics,
    }
    save_metadata(metadata, MODEL_METADATA_PATH)


if __name__ == "__main__":
    main()


# TODO: Add Regularization
# TODO: Add Optimizers - Added Adam Optimizer - Done



# LOGS
# Has stuck here - Train and Loss high - Resolved this by removing the reset_gradient which was called twice.
# Epoch 11/100 - Train Loss: 103.124531 - Val Loss: 39.469859
# Epoch 12/100 - Train Loss: 103.130404 - Val Loss: 39.472441
# Epoch 13/100 - Train Loss: 103.132880 - Val Loss: 39.473489
# Epoch 14/100 - Train Loss: 103.133924 - Val Loss: 39.473910
# Epoch 15/100 - Train Loss: 103.134365 - Val Loss: 39.474077
# Epoch 16/100 - Train Loss: 103.134551 - Val Loss: 39.474142
# Epoch 17/100 - Train Loss: 103.134629 - Val Loss: 39.474167
# Epoch 18/100 - Train Loss: 103.134663 - Val Loss: 39.474176
# Epoch 19/100 - Train Loss: 103.134677 - Val Loss: 39.474179
# Epoch 20/100 - Train Loss: 103.134682 - Val Loss: 39.474180
# Epoch 21/100 - Train Loss: 103.134685 - Val Loss: 39.474181

# Changed lookback window from 60 -> 30
# Model became really slow and did not converge

# Changed lookback window from 30 -> 90
# Its converging really good for now
# Checkpoint saved at epoch 36 with val loss 20.237960
# Epoch 37/100 - Train Loss: 19.289976 - Val Loss: 20.029927
# Model saved to ./data/checkpoints/checkpoint.pkl
# Checkpoint saved at epoch 37 with val loss 20.029927
#...
# Checkpoint saved at epoch 88 with val loss 13.376115
# Epoch 89/100 - Train Loss: 14.322639 - Val Loss: 13.349268
# Model saved to ./data/checkpoints/checkpoint.pkl
# Checkpoint saved at epoch 89 with val loss 13.349268