LOOKBACK_WINDOW = 60  # Use 60 days of history to predict the next day
TEST_SPLIT = 0.15     # 15% of data for testing
VAL_SPLIT = 0.15      # 15% for validation

EPOCHS = 100
LEARNING_RATE = 0.001


# project/
# │
# ├── data_utils.py         # All data preprocessing utilities
# ├── lstm_cell.py          # LSTMCell class
# ├── lstm_layer.py         # LSTMLayer class
# ├── simple_layer.py       # Linear layer
# ├── train.py              # Training script (entry point)
# ├── evaluate.py           # Evaluation script
# └── config.py             # Hyperparameters