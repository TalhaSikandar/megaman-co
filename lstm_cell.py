import numpy as np  

# From LSTM paper:
# f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)       # Forget gate
# i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)       # Input gate
# o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)       # Output gate
# g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)          # Cell candidate

# c_t = f_t * c_{t-1} + i_t * g_t                 # Cell state update
# h_t = o_t * tanh(c_t)                           # Hidden state output

# Architecture:
# Input: 6 features (normalized price, open, high, low, volume, change)
# Output: 1 target (raw price)
    # Input (60, 6) →
    # LSTM (hidden_size=64) →
    # Last hidden state (h_60) →
    # Fully connected layer (64 → 1) →
    # Output: predicted price


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Concatenated weights: [x_t, h_{t-1}] @ W + b
        concat_size = input_size + hidden_size

        # Xavier initialization
        limit = np.sqrt(1 / concat_size)
        self.W_f = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_f = np.zeros((hidden_size, 1))

        self.W_i = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_i = np.zeros((hidden_size, 1))

        self.W_c = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_c = np.zeros((hidden_size, 1))

        self.W_o = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_o = np.zeros((hidden_size, 1))

        # Adam optimizer states
        self.m = {}
        self.v = {}
        self.t = 0  # timestep

        for name in ['W_f', 'b_f', 'W_i', 'b_i', 'W_o', 'b_o', 'W_c', 'b_c']:
            param = getattr(self, name)
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def forward(self, x_t, h_prev, c_prev):
        concat = np.vstack((h_prev, x_t))  # (input_size + hidden_size, 1)

        self.f_t = self._sigmoid(np.dot(self.W_f, concat) + self.b_f)
        self.i_t = self._sigmoid(np.dot(self.W_i, concat) + self.b_i)
        self.o_t = self._sigmoid(np.dot(self.W_o, concat) + self.b_o)
        self.g_t = np.tanh(np.dot(self.W_c, concat) + self.b_c)

        self.c_prev = c_prev
        self.c_t = self.f_t * c_prev + self.i_t * self.g_t
        self.h_t = self.o_t * np.tanh(self.c_t)
        self.concat = concat  # Cache input for backward

        return self.h_t, self.c_t

    def backward(self, dh_next, dc_next):
        do_t = dh_next * np.tanh(self.c_t)
        dc_t = dh_next * self.o_t * (1 - np.tanh(self.c_t) ** 2) + dc_next

        df_t = dc_t * self.c_prev
        di_t = dc_t * self.g_t
        dg_t = dc_t * self.i_t
        dc_prev = dc_t * self.f_t

        df_t_raw = df_t * self.f_t * (1 - self.f_t)
        di_t_raw = di_t * self.i_t * (1 - self.i_t)
        do_t_raw = do_t * self.o_t * (1 - self.o_t)
        dg_t_raw = dg_t * (1 - self.g_t ** 2)

        dW_f = np.dot(df_t_raw, self.concat.T)
        dW_i = np.dot(di_t_raw, self.concat.T)
        dW_o = np.dot(do_t_raw, self.concat.T)
        dW_c = np.dot(dg_t_raw, self.concat.T)

        db_f = df_t_raw
        db_i = di_t_raw
        db_o = do_t_raw
        db_c = dg_t_raw

        self.dW_f += dW_f
        self.db_f += db_f
        self.dW_i += dW_i
        self.db_i += db_i
        self.dW_o += dW_o
        self.db_o += db_o
        self.dW_c += dW_c
        self.db_c += db_c

        d_concat = (
            np.dot(self.W_f.T, df_t_raw) +
            np.dot(self.W_i.T, di_t_raw) +
            np.dot(self.W_o.T, do_t_raw) +
            np.dot(self.W_c.T, dg_t_raw)
        )

        dh_prev = d_concat[:self.hidden_size, :]
        dx_t = d_concat[self.hidden_size:, :]

        return dx_t, dh_prev, dc_prev


    
    def reset_gradients(self):
            self.dW_f = np.zeros_like(self.W_f)
            self.db_f = np.zeros_like(self.b_f)
            self.dW_i = np.zeros_like(self.W_i)
            self.db_i = np.zeros_like(self.b_i)
            self.dW_o = np.zeros_like(self.W_o)
            self.db_o = np.zeros_like(self.b_o)
            self.dW_c = np.zeros_like(self.W_c)
            self.db_c = np.zeros_like(self.b_c)

    def update_weights(self, lr):
            # Gradient descent
            self.W_f -= lr * self.dW_f
            self.b_f -= lr * self.db_f
            self.W_i -= lr * self.dW_i
            self.b_i -= lr * self.db_i
            self.W_o -= lr * self.dW_o
            self.b_o -= lr * self.db_o
            self.W_c -= lr * self.dW_c
            self.b_c -= lr * self.db_c

            # Reset gradients after update
            self.reset_gradients()

    def update_weights_adam(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1

        for name in ['W_f', 'b_f', 'W_i', 'b_i', 'W_o', 'b_o', 'W_c', 'b_c']:
            param = getattr(self, name)
            grad = getattr(self, 'd' + name)

            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grad ** 2)

            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)

            param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

            setattr(self, name, param)

        self.reset_gradients()


    @staticmethod  
    def _sigmoid(x):  
        return 1 / (1 + np.exp(-x))  
    

    

    
