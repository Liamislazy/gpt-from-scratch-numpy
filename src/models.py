import numpy as np
from layers import Linear


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SingleHeadAttention:
    def __init__(self, d_model: int, d_k: int):
        self.d_k = d_k

        self.W_q = np.random.randn(d_model, d_k) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_k) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_k) / np.sqrt(d_model)

    def forward(self, X : np.ndarray) -> np.ndarray:
        # X shape: (batch_size, seq_len, d_model)

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        S = (Q @ K.transpose((0, 2, 1))) / np.sqrt(self.d_k)
        W = softmax(S)

        output = W @ V

        return output

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_k = d_model // num_heads
        
        # 1. Initialize self.heads (list of SingleHeadAttention instances)
        self.heads = [SingleHeadAttention(d_model, self.d_k) for _ in range(num_heads)]
        # 2. Initialize self.W_o
        self.W_o = Linear(num_heads * self.d_k, d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # 1. Run X through each head
        output_heads = [head.forward(X) for head in self.heads]

        # 2. Concatenate head outputs along axis=-1
        concat = np.concatenate(output_heads, axis=-1)

        # 3. Project with W_o
        output = self.W_o.forward(concat)

        return output