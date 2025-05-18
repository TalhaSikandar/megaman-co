import numpy as np
import argparse
import os
from config import LOOKBACK_WINDOW, TEST_SPLIT, VAL_SPLIT
from metadata import (
    PREPROCESSED_DATA_PATH,
)

def parse_volume(vol_str):
    """Parse volume string with K/M suffixes to float."""
    vol_str = vol_str.strip("\"")
    if vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1e6
    elif vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1e3
    else:
        return float(vol_str)

def normalize(x):
    """Min-max normalization."""
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def create_sequences(data, targets, window_size=60):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y).reshape(-1, 1)

def split_data(X, y, test_split=0.2, val_split=0.1):
    """Split data into train, validation, and test sets."""
    num_samples = len(X)
    train_end = int(num_samples * (1 - test_split - val_split))
    val_end = train_end + int(num_samples * val_split)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_data(output_path, **arrays):
    """Save arrays to a .npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Data saved to: {output_path}")
    for k, v in arrays.items():
        print(f"{k}: {v.shape}")

def load_data(npz_path):
    """Load preprocessed data from .npz file."""
    data = np.load(npz_path)
    return (data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'])

def preprocess(input_csv, output_path,
               lookback_window=LOOKBACK_WINDOW,
               test_split=TEST_SPLIT,
               val_split=VAL_SPLIT):
    """Full preprocessing pipeline from CSV to .npz."""
    data = np.genfromtxt(input_csv, delimiter=',', dtype=str, skip_header=1)
    print("CSV Loaded. Shape:", data.shape)

    prices = np.array([float(x.strip("\"")) for x in data[:, 1]])
    opens_ = np.array([float(x.strip("\"")) for x in data[:, 2]])
    highs = np.array([float(x.strip("\"")) for x in data[:, 3]])
    lows = np.array([float(x.strip("\"")) for x in data[:, 4]])
    volumes = np.array([parse_volume(x) for x in data[:, 5]])
    changes = np.array([float(x.strip("\"").strip('%')) for x in data[:, 6]])

    norm_price = normalize(prices)
    norm_open = normalize(opens_)
    norm_high = normalize(highs)
    norm_low = normalize(lows)
    norm_volume = normalize(volumes)
    norm_change = normalize(changes)

    features = np.column_stack([
        norm_price, norm_open, norm_high,
        norm_low, norm_volume, norm_change
    ])

    # Use raw price for prediction target
    targets = prices[1:]
    X, y = create_sequences(features[:-1], targets, lookback_window)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, test_split, val_split
    )

    save_data(output_path,
              X_train=X_train, y_train=y_train,
              X_val=X_val, y_val=y_val,
              X_test=X_test, y_test=y_test)

def main():
    parser = argparse.ArgumentParser(description="Preprocess stock data for LSTM prediction.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_path", type=str, default=PREPROCESSED_DATA_PATH, help="Output .npz file path.")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_WINDOW, help="Lookback window size.")
    parser.add_argument("--test_split", type=float, default=TEST_SPLIT, help="Test split ratio.")
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT, help="Validation split ratio.")
    args = parser.parse_args()

    preprocess(args.input_csv, 
               args.output_path,
               lookback_window=args.lookback,
               test_split=args.test_split,
               val_split=args.val_split)

if __name__ == "__main__":
    main()

#python preprocess.py --input_csv habib_bank_21.csv --output_path data/processed_21.npz
