import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image
import os

# Wczytanie MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32)  / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# One-hot
def to_one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    for i, val in enumerate(y):
        oh[i, val] = 1.0
    return oh

y_train_oh = to_one_hot(y_train)
y_test_oh  = to_one_hot(y_test)

# Funkcje
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(z.dtype)

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# MLP
class MLP_2Hidden:
    def __init__(self, input_dim=784, hidden_dim1=128, hidden_dim2=64, output_dim=10, lr=0.01):
        limit1 = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim1).astype(np.float32) * limit1
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)

        limit2 = np.sqrt(2.0 / hidden_dim1)
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2).astype(np.float32) * limit2
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)

        limit3 = np.sqrt(2.0 / hidden_dim2)
        self.W3 = np.random.randn(hidden_dim2, output_dim).astype(np.float32) * limit3
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

        self.lr = lr

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        y_pred = softmax(z3)
        return y_pred, (X, z1, a1, z2, a2, z3, y_pred)

    def backward(self, cache, y_true):
        X, z1, a1, z2, a2, z3, y_pred = cache
        bs = X.shape[0]

        dz3 = (y_pred - y_true) / bs
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_batch(self, X_batch, y_batch):
        y_pred, cache = self.forward(X_batch)
        loss = cross_entropy_loss(y_pred, y_batch)
        acc  = accuracy(y_pred, y_batch)
        self.backward(cache, y_batch)
        return loss, acc

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

# Trenowanie
batch_size = 128
epochs = 10
lr = 0.01
model = MLP_2Hidden(lr=lr)
num_train = x_train.shape[0]
num_batches = num_train // batch_size

for e in range(epochs):
    idx = np.arange(num_train)
    np.random.shuffle(idx)
    epoch_loss = 0
    epoch_acc  = 0

    for i in range(num_batches):
        b_idx = idx[i*batch_size:(i+1)*batch_size]
        Xb = x_train[b_idx]
        Yb = y_train_oh[b_idx]
        l, a = model.train_batch(Xb, Yb)
        epoch_loss += l
        epoch_acc  += a

    print(f"Epoch {e+1}/{epochs} - loss: {epoch_loss/num_batches:.4f}, accuracy: {epoch_acc/num_batches:.4f}")

# Test
y_pred_test, _ = model.forward(x_test)
test_loss = cross_entropy_loss(y_pred_test, y_test_oh)
test_acc  = accuracy(y_pred_test, y_test_oh)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')

        img = img.resize((28, 28), Image.LANCZOS)

        img_array = np.array(img)

        img_array = 255 - img_array

        img_array = img_array.astype(np.float32) / 255.0

        img_vector = img_array.flatten().reshape(1, -1)

        return img_vector
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazka: {e}")
        return None


def test_custom_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Plik {image_path} nie istnieje!")
        return

    X_custom = preprocess_image(image_path)

    if X_custom is None:
        print("Nie udało się przetworzyć obrazka.")
        return

    y_pred = model.predict(X_custom)
    pred_label = np.argmax(y_pred, axis=1)[0]

    img = Image.open(image_path).convert('L').resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)
    img_array = 255 - img_array

    plt.imshow(img_array, cmap='gray')
    plt.title(f"Przewidywana etykieta: {pred_label}")
    plt.axis('off')
    plt.show()


my_image_path = "zero.png"
test_custom_image(model, my_image_path)

my_image_path = "jeden.png"
test_custom_image(model, my_image_path)

my_image_path = "dwa.png"
test_custom_image(model, my_image_path)

my_image_path = "trzy.png"
test_custom_image(model, my_image_path)

my_image_path = "cztery.png"
test_custom_image(model, my_image_path)

my_image_path = "piec.png"
test_custom_image(model, my_image_path)

my_image_path = "szesc.png"
test_custom_image(model, my_image_path)

my_image_path = "siedem.png"
test_custom_image(model, my_image_path)

my_image_path = "osiem.png"
test_custom_image(model, my_image_path)

my_image_path = "dziewiec.png"
test_custom_image(model, my_image_path)