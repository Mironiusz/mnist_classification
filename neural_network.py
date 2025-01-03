import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    # Stabilizowana wersja softmax
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

class NeuralNetwork:
    """
    Prosty MLP z dwiema warstwami ukrytymi do klasyfikacji cyfr MNIST.
    """
    def __init__(self, input_dim=784, hidden_dim1=128, hidden_dim2=64, output_dim=10, lr=0.01):
        """
        Inicjalizacja wag metodą 'Xavier/He initialization' (w uproszczeniu).
        """
        # Warstwa 1
        limit1 = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim1).astype(np.float32) * limit1
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)

        # Warstwa 2
        limit2 = np.sqrt(2.0 / hidden_dim1)
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2).astype(np.float32) * limit2
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)

        # Warstwa 3 (wyjściowa)
        limit3 = np.sqrt(2.0 / hidden_dim2)
        self.W3 = np.random.randn(hidden_dim2, output_dim).astype(np.float32) * limit3
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

        # Parametr uczenia
        self.lr = lr

    def forward(self, X):
        """
        Wykonanie przepływu w przód (forward pass).
        Zwraca (y_pred, (cache)), gdzie:
          - y_pred to wynik softmax (predykcja)
          - cache to krotka z wewn. wartościami potrzebnymi w backprop
        """
        # Warstwa 1
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        # Warstwa 2
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)

        # Warstwa 3 (wyjściowa)
        z3 = a2 @ self.W3 + self.b3
        y_pred = softmax(z3)

        # Zapisujemy potrzebne wartości w cache do backprop
        cache = (X, z1, a1, z2, a2, z3, y_pred)
        return y_pred, cache

    def backward(self, cache, y_true):
        """
        Wykonanie przepływu wstecz (backpropagation) i aktualizacja wag.
        """
        X, z1, a1, z2, a2, z3, y_pred = cache
        bs = X.shape[0]  # rozmiar batcha

        # Gradient względem z3
        dz3 = (y_pred - y_true) / bs
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Gradient względem warstwy 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Gradient względem warstwy 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Aktualizacja wag
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_batch(self, X_batch, y_batch):
        """
        Trenuje jedną partię (batch) danych.
        Zwraca (loss, accuracy) dla tej partii.
        """
        y_pred, cache = self.forward(X_batch)
        loss = cross_entropy_loss(y_pred, y_batch)
        acc = accuracy(y_pred, y_batch)

        # Backprop
        self.backward(cache, y_batch)
        return loss, acc

    def predict(self, X):
        """
        Zwraca tablicę prawdopodobieństw predykcji (softmax).
        """
        y_pred, _ = self.forward(X)
        return y_pred
