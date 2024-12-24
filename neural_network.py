import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Inicjalizacja wag i biasów
        self.weights = [
            np.random.uniform(-0.1, 0.1, (28**2, 128)),  # Wejście do pierwszej warstwy ukrytej
            np.random.uniform(-0.1, 0.1, (128, 64)),     # Pierwsza warstwa ukryta do drugiej warstwy ukrytej
            np.random.uniform(-0.1, 0.1, (64, 10))       # Druga warstwa ukryta do warstwy wyjściowej
        ]
        self.biases = [
            np.zeros(128),
            np.zeros(64),
            np.zeros(10)
        ]

    def activation_relu(self, x):
        return np.maximum(0, x)

    def activation_softmax(self, x):
        exp_shifted = np.exp(x - np.max(x))  # Stabilizacja
        return exp_shifted / np.sum(exp_shifted)

    def forward_pass(self, x):
        activations = [x]
        z_values = []

        # Warstwy ukryte
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], w) + b
            a = self.activation_relu(z)
            z_values.append(z)
            activations.append(a)

        # Warstwa wyjściowa
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.activation_softmax(z)
        z_values.append(z)
        activations.append(a)

        return activations, z_values

    def train(self, x_train, y_train, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                # Normalizacja wejścia
                x = x.flatten() / 255.0

                # Forward pass
                activations, z_values = self.forward_pass(x)

                # Obliczenie strat (cross-entropy)
                loss = -np.log(activations[-1][y] + 1e-15)  # Dodanie małej wartości dla stabilności
                total_loss += loss

                # Backward pass (prosta implementacja bez batchów)
                # Gradient dla warstwy wyjściowej
                delta = activations[-1].copy()
                delta[y] -= 1  # Gradient cross-entropy z softmax

                # Aktualizacja wag i biasów
                for i in reversed(range(len(self.weights))):
                    a_prev = activations[i]
                    dw = np.outer(a_prev, delta)
                    db = delta
                    self.weights[i] -= learning_rate * dw
                    self.biases[i] -= learning_rate * db

                    if i != 0:
                        delta = np.dot(delta, self.weights[i].T)
                        delta = delta * (z_values[i-1] > 0)  # Gradient ReLU

            average_loss = total_loss / len(x_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}")

    def predict(self, x):
        activations, _ = self.forward_pass(x.flatten() / 255.0)
        return np.argmax(activations[-1])

# Przykładowe użycie
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    nn = NeuralNetwork()
    nn.train(x_train, y_train, epochs=10, learning_rate=0.01)

    # Przykładowa predykcja
    print("Predykcja:", nn.predict(x_test[0]))
    print("Prawdziwa etykieta:", y_test[0])
