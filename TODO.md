# TODOs

## Data Preparation
1. **Feature Engineering**
   - Combine normalized features into a single matrix `(samples, features)`
   - Add any additional features (e.g., moving averages, technical indicators)
   - Handle any remaining NaN/infinite values

2. **Sequence Creation**
   - Define lookback window size (e.g., 60 days)
   - Create sliding window sequences `(num_sequences, lookback_window, num_features)`
   - Create corresponding target values (next day's price)

3. **Train/Val/Test Split**
   - Split chronologically (e.g., 70% train, 15% val, 15% test)
   - Ensure no data leakage between sets

## LSTM Implementation
4. **LSTM Cell Components**
   - Implement forget gate (sigmoid)
   - Implement input gate (sigmoid + tanh)
   - Implement output gate (sigmoid + tanh)
   - Implement cell state update
   - Implement hidden state update

5. **Network Architecture**
   - Initialize parameters (weights, biases)
   - Implement forward pass through LSTM layer
   - Implement dense output layer
   - Implement prediction function

## Training Infrastructure
6. **Loss & Optimization**
   - Implement MSE loss function
   - Implement gradient calculation via BPTT
   - Implement Adam optimizer
   - Add L2 regularization

7. **Training Loop**
   - Implement mini-batch training
   - Add validation set evaluation
   - Implement early stopping
   - Add loss logging/visualization

## Evaluation
8. **Performance Metrics**
   - Implement RMSE, MAE, MAPE
   - Create prediction vs actual plots
   - Backtest trading strategy (optional)

## Code Structure
9. **Modular Implementation**
   - `data_preprocessor.py` - All data handling
   - `lstm.py` - Core LSTM implementation
   - `train.py` - Training infrastructure
   - `evaluate.py` - Metrics and visualization
   - `config.py` - Hyperparameters and constants

