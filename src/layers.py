import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # using He(Kaiming) initialization because i'm using ReLU : sqrt(2/n_in)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.B = np.random.randn(1, output_size)

        # cache for backprop
        self.dW = None
        self.db = None
        self.dX = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.B

    def backward(self, dY):
        self.dW = self.X.T @ dY
        self.db = np.sum(dY, axis=0, keepdims=True)
        self.dX = dY @ self.W.T

        return self.dW, self.db, self.dX

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pos_arr = np.arange(seq_length)[:, np.newaxis]

    div_term = 1 / (10000 ** (np.arange(0, d_model, 2) / d_model))

    pos_seq = np.zeros((seq_length, d_model))

    pos_seq[:, 0::2] = np.sin(pos_arr * div_term)
    pos_seq[:, 1::2] = np.cos(pos_arr * div_term)
    
    return pos_seq


class LayerNorm():
    def __init__(self, normalized_shape, eps = 1e-5):
        if not isinstance(normalized_shape, tuple):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = normalized_shape

        self.normalized_shape = normalized_shape
        self.eps = eps

        # learnable param
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        
        scaled_x = (X - mean) / np.sqrt(variance + self.eps)
        return self.gamma * scaled_x + self.beta




