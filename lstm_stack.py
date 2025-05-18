import numpy as np
from lstm_cell import LSTMCell
from simple_layer import Linear
from simple_layer import Dropout, ReLU  

class StackedLSTMLayer:
    def __init__(self, input_size, layer_configs, output_size):
        """
        layer_configs: list of dicts, each with keys:
            - 'hidden_size': int
            - 'dropout': float (0.0-1.0)
        Example:
            [
                {'hidden_size': 50, 'dropout': 0.2},
                {'hidden_size': 60, 'dropout': 0.3},
                {'hidden_size': 80, 'dropout': 0.4},
                {'hidden_size': 120, 'dropout': 0.5},
            ]
        """
        self.layers = []
        self.dropouts = []
        self.activations = []
        prev_size = input_size
        for cfg in layer_configs:
            self.layers.append(LSTMCell(prev_size, cfg['hidden_size']))
            # print(cfg['hidden_size'])
            self.activations.append(ReLU())
            self.dropouts.append(Dropout(cfg['dropout']))
            prev_size = cfg['hidden_size']
        self.fc = Linear(prev_size, output_size)
        self.losses = []
        self.val_losses = []
        self.learning_rate = 0.001
        self.best_val_loss = float('inf')

    def forward(self, input_sequence, training=True):
        self.h = [np.zeros((layer.hidden_size, 1)) for layer in self.layers]
        self.c = [np.zeros((layer.hidden_size, 1)) for layer in self.layers]
        self.inputs = []
        self.hiddens = [[] for _ in self.layers]
        self.cells = [[] for _ in self.layers]
        self.activ_outs = [[] for _ in self.layers]
        self.drop_masks = [[] for _ in self.layers]  # Store mask per time step
        seq_len = input_sequence.shape[0]
        x = None
        for t in range(seq_len):
            x = input_sequence[t].reshape(-1, 1)
            self.inputs.append(x)
            for i, (layer, activation, dropout) in enumerate(zip(self.layers, self.activations, self.dropouts)):
                self.h[i], self.c[i] = layer.forward(x, self.h[i], self.c[i])
                self.hiddens[i].append(self.h[i])
                self.cells[i].append(self.c[i])
                x = activation.forward(self.h[i])
                self.activ_outs[i].append(x)
                if training:
                    x = dropout.forward(x, training=True)
                    self.drop_masks[i].append(dropout.mask.copy())  # Save mask for this time step/layer
                else:
                    x = dropout.forward(x, training=False)
                    self.drop_masks[i].append(np.ones_like(x))
        self.y_pred = self.fc.forward(x)
        return self.y_pred
    def compute_loss(self, y_true):
        return np.mean((self.y_pred - y_true) ** 2)

    def backward(self, y_true, learning_rate=0.001, clip_value=5.0):
        # 1) Compute top‑level loss gradient (MSE)
        dy = self.y_pred - y_true              # shape: (output_size, 1)

        # 2) Backprop through the final FC layer
        dL_dh_top = self.fc.backward(dy)       # shape: (hidden_N, 1)

        # 3) Prepare per‑layer hidden/cell gradients
        n_layers = len(self.layers)
        dL_dh_next = [np.zeros((L.hidden_size, 1)) for L in self.layers]
        dL_dc_next = [np.zeros((L.hidden_size, 1)) for L in self.layers]
        # Seed the top layer’s hidden grad at the last time‑step
        dL_dh_next[-1] = dL_dh_top.copy()

        # 4) Reset all parameter gradients before accumulating
        self.fc.reset_gradients()
        for L in self.layers:
            L.reset_gradients()

        seq_len = len(self.inputs)

        # 5) Backpropagate through time
        for t in reversed(range(seq_len)):
            # For each layer, go from top → bottom
            for i in reversed(range(n_layers)):
                # 5.1) Start with that layer’s hidden gradient
                dh = dL_dh_next[i]              # shape: (hidden_i, 1)
                dc = dL_dc_next[i]              # shape: (hidden_i, 1)

                # 5.2) Undo dropout
                mask = self.drop_masks[i][t]    # shape: (hidden_i, 1)
                dh = dh * mask / (1.0 - self.dropouts[i].dropout_rate)

                # 5.3) Backprop through ReLU
                dh = self.activations[i].backward(dh)

                # 5.4) Backprop through the LSTM cell
                #    backward returns: dx (to feed lower layer), dh_prev, dc_prev
                dx, dh_prev, dc_prev = self.layers[i].backward(dh, dc)

                # 5.5) Clip each gradient if desired
                dx      = np.clip(dx,      -clip_value, clip_value)
                dh_prev = np.clip(dh_prev, -clip_value, clip_value)
                dc_prev = np.clip(dc_prev, -clip_value, clip_value)

                # 5.6) Store next‑time‑step gradients for this layer
                dL_dh_next[i] = dh_prev
                dL_dc_next[i] = dc_prev

                # 5.7) Pass “dx” down to the next lower layer
                if i > 0:
                    dL_dh_next[i-1] += dx

            # end for each layer
        # end for each time step

        # 6) Finally, update all parameters via Adam
        self.fc.update_weights_adam(learning_rate)
        for L in self.layers:
            L.update_weights_adam(learning_rate)


    def reset_state(self):
        self.h = [np.zeros((layer.hidden_size, 1)) for layer in self.layers]
        self.c = [np.zeros((layer.hidden_size, 1)) for layer in self.layers]

"""
StackedLSTMLayer Architecture:

Input Sequence (T x input_size)
        │
        ▼
┌────────────────────────┐
│   LSTM Layer 1         │
│   (input_size → h1)    │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Activation (ReLU)      │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Dropout (e.g. 0.2)      │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│   LSTM Layer 2         │
│   (h1 → h2)            │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Activation (ReLU)      │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Dropout (e.g. 0.3)      │
└────────────────────────┘
        │
        ▼
       ...
        ▼
┌────────────────────────┐
│   LSTM Layer N         │
│   (h_{N-1} → hN)       │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Activation (ReLU)      │
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Dropout (e.g. 0.5)      │
└────────────────────────┘
        │
        ▼
Last hidden state (hN_T)
        │
        ▼
┌────────────────────────┐
│ Fully Connected Layer  │
│   (hN → output_size)   │
└────────────────────────┘
        │
        ▼
      Output
"""
