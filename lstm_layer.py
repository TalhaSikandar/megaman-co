import numpy as np
from lstm_cell import LSTMCell
from simple_layer import Linear
from config import LEARNING_RATE

# Input Sequence (shape: [lookback_window, 6 features])
#         │
#         ▼
# ┌─────────────────────────────┐
# │        LSTM Layer           │
# │  (hidden_size = 64 units)   │
# └─────────────────────────────┘
#         │
#         ▼
# ┌─────────────────────────────┐
# │    Linear (Fully Connected) │
# │      (output_size = 1)      │
# └─────────────────────────────┘
#         │
#         ▼
#   Predicted Next-Day Price

class LSTMLayer:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.fc = Linear(hidden_size, output_size)
        self.epochs = 0
        self.current_epoch = 0
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.learning_rate = LEARNING_RATE
        self.best_val_loss = float('inf')


    def forward(self, input_sequence):
        self.h_t = np.zeros((self.hidden_size, 1))
        self.c_t = np.zeros((self.hidden_size, 1))
        self.inputs = []
        self.hiddens = []
        self.cells = []

        for t in range(input_sequence.shape[0]):
            x_t = input_sequence[t].reshape(-1, 1)
            self.inputs.append(x_t)
            self.h_t, self.c_t = self.cell.forward(x_t, self.h_t, self.c_t)
            self.hiddens.append(self.h_t)
            self.cells.append(self.c_t)

        self.y_pred = self.fc.forward(self.h_t)
        return self.y_pred

    def backward(self, y_true, learning_rate=0.001):
        dy = self.y_pred - y_true

        dL_dh = self.fc.backward(dy)  # Gradient w.r.t. h_t

        dL_dh_next = dL_dh
        dL_dc_next = np.zeros((self.hidden_size, 1))

        self.cell.reset_gradients()  # Reset before BPTT
        self.fc.reset_gradients() # Reset before BPTT

        for t in reversed(range(len(self.inputs))):
            h_prev = self.hiddens[t-1] if t > 0 else np.zeros_like(self.hiddens[0])
            c_prev = self.cells[t-1] if t > 0 else np.zeros_like(self.cells[0])
            dx, dL_dh_next, dL_dc_next = self.cell.backward(
                dL_dh_next, dL_dc_next
            )

        # self.fc.W -= learning_rate * self.fc.dW
        # self.fc.b -= learning_rate * self.fc.db
        self.fc.update_weights_adam(learning_rate)
        self.cell.update_weights_adam(learning_rate)

    def compute_loss(self, y_true):
        return np.mean((self.y_pred - y_true) ** 2)