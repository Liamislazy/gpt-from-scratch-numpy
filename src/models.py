import numpy as np
from src.layers import Linear, Dropout, GELU, softmax, derived_GELU


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


class FeedForward:
    def __init__(self, d_model, d_ff, dropout: float = 0.1):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        self.hidden = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.hidden = self.linear1.forward(X)
        out = self.linear2.forward(GELU(self.hidden))
        return self.dropout.forward(out, training=training)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.dropout.backward(grad)
        _, _, grad = self.linear2.backward(grad)
        grad = grad * derived_GELU(self.hidden)
        _, _, grad = self.linear1.backward(grad)
        return grad

class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad.fill(0.0)

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    
class Block(Module):
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass