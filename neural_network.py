import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
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
    Ogólny MLP z parametryzowaną liczbą warstw, neuronów i funkcji aktywacji.
    """
    def __init__(self, layers, activations, lr=0.01, loss_fn=cross_entropy_loss, accuracy_fn=accuracy):
        """
        Parametry:
        - layers: lista określająca liczbę neuronów w kolejnych warstwach,
                  np. [784, 128, 64, 10]
        - activations: lista funkcji aktywacji dla warstw ukrytych,
                       np. [relu, relu] dla dwóch warstw ukrytych
        - lr: współczynnik uczenia
        - loss_fn: funkcja straty
        - accuracy_fn: funkcja mierząca dokładność
        """
        self.lr = lr
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn

        self.num_layers = len(layers) - 1  # liczba warstw z wagami (bez wejścia)
        self.weights = []
        self.biases = []
        self.activations = activations
        self.activation_derivatives = []

        # Inicjalizacja wag i biasów dla każdej warstwy
        for i in range(self.num_layers):
            limit = np.sqrt(2.0 / layers[i])
            self.weights.append(np.random.randn(layers[i], layers[i+1]).astype(np.float32) * limit)
            self.biases.append(np.zeros((1, layers[i+1]), dtype=np.float32))
            # Dla wyjściowej warstwy nie przypisujemy funkcji aktywacji z listy,
            # gdyż użyjemy softmax w forward
            if i < self.num_layers - 1:
                # Zakładamy, że wszystkie warstwy ukryte używają podanej funkcji (np. ReLU)
                # i jej pochodnej
                self.activation_derivatives.append(relu_derivative)  # można rozszerzyć do innych

    def forward(self, X):
        """
        Forward pass przez wszystkie warstwy.
        """
        cache = {}
        cache['a0'] = X
        # Przejście przez warstwy ukryte
        for i in range(self.num_layers - 1):
            z = cache[f'a{i}'] @ self.weights[i] + self.biases[i]
            cache[f'z{i+1}'] = z
            # Użycie ReLU dla warstw ukrytych (można rozszerzyć do innych aktywacji)
            cache[f'a{i+1}'] = relu(z)
        # Warstwa wyjściowa
        z_out = cache[f'a{self.num_layers - 1}'] @ self.weights[-1] + self.biases[-1]
        cache[f'z{self.num_layers}'] = z_out
        y_pred = softmax(z_out)
        cache[f'a{self.num_layers}'] = y_pred
        return y_pred, cache

    def backward(self, cache, y_true):
        """
        Backpropagation dla całej sieci.
        """
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        bs = cache['a0'].shape[0]
        # Gradient dla warstwy wyjściowej
        delta = (cache[f'a{self.num_layers}'] - y_true) / bs
        # Obliczenia dla ostatniej warstwy
        a_prev = cache[f'a{self.num_layers - 1}']
        grads_W[-1] = a_prev.T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Backprop przez warstwy ukryte
        for i in range(self.num_layers - 2, -1, -1):
            dz = (delta @ self.weights[i+1].T) * relu_derivative(cache[f'z{i+1}'])
            a_prev = cache[f'a{i}']
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = np.sum(dz, axis=0, keepdims=True)
            delta = dz

        # Aktualizacja wag i biasów
        for i in range(self.num_layers):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i]  -= self.lr * grads_b[i]

    def train_batch(self, X_batch, y_batch):
        y_pred, cache = self.forward(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        acc = self.accuracy_fn(y_pred, y_batch)
        self.backward(cache, y_batch)
        return loss, acc

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred
