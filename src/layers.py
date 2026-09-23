import numpy as np
import math

# use GELU because it is better
_erf_vectorized = np.vectorize(math.erf)
def GELU(x: np.ndarray) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + _erf_vectorized(x / np.sqrt(2.0)))

def derived_GELU(x):
    x = np.asarray(x, dtype=np.float64)
    
    cdf = 0.5 * (1.0 + _erf_vectorized(x / np.sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.square(x))
    
    return cdf + x * pdf

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def ReLU(X):
    return np.maximum(X, 0)

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
        flat_x = self.X.reshape(-1, self.input_size)
        flat_dy = dY.reshape(-1, self.output_size)

        self.dW = flat_x.T @ flat_dy
        self.db = np.sum(flat_dy, axis=0, keepdims=True)
        self.dX = (flat_dy @ self.W.T).reshape(self.X.shape)

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


class LayerNorm:
    def __init__(self, normalized_shape, eps: float = 1e-5):
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


class Dropout:
    def __init__(self, p: float):
        """Initialize the dropout layer.

        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.

        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if not training:
            self.mask = None
            return x

        self.mask = np.random.binomial(n=1, p=1 - self.p, size=x.shape)
        return (x * self.mask) / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.

        Use the stored self.mask from the most recent forward pass.
        """
        if self.mask is None:
            return grad

        return (grad * self.mask) / (1 - self.p)

