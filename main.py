import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork, cross_entropy_loss, accuracy, relu

(x_full, y_full), (x_test, y_test) = mnist.load_data()

x_full = x_full.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_full = x_full.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Etykieta: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

def to_one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    for i, val in enumerate(y):
        oh[i, val] = 1.0
    return oh

y_train_oh = to_one_hot(y_train)
y_val_oh   = to_one_hot(y_val)
y_test_oh  = to_one_hot(y_test)

layers = [784, 128, 64, 10]
activations = [relu, relu]
model = NeuralNetwork(layers=layers, activations=activations, lr=0.1)

batch_size = 128
epochs = 100
num_train_samples = x_train.shape[0]
num_batches = num_train_samples // batch_size

for e in range(epochs):
    idx = np.arange(num_train_samples)
    np.random.shuffle(idx)
    epoch_loss = 0.0
    epoch_acc  = 0.0

    for i in range(num_batches):
        b_idx = idx[i*batch_size : (i+1)*batch_size]
        Xb = x_train[b_idx]
        Yb = y_train_oh[b_idx]

        loss, acc = model.train_batch(Xb, Yb)
        epoch_loss += loss
        epoch_acc  += acc

    print(f"Epoch {e+1}/{epochs} - loss: {epoch_loss / num_batches:.4f}, accuracy: {epoch_acc / num_batches:.4f}")

    y_pred_val = model.predict(x_val)
    val_loss = cross_entropy_loss(y_pred_val, y_val_oh)
    val_acc  = accuracy(y_pred_val, y_val_oh)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

y_pred_test = model.predict(x_test)
test_loss = cross_entropy_loss(y_pred_test, y_test_oh)
test_acc  = accuracy(y_pred_test, y_test_oh)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


# Funkcje do wczytywania naszych cyferek

def center_and_pad_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img.thumbnail((20, 20), Image.LANCZOS)
        img_array = np.array(img)
        img_array = 255 - img_array
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        if not rows.any() or not cols.any():
            return img_array
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        digit = img_array[y_min:y_max+1, x_min:x_max+1]
        digit_height, digit_width = digit.shape
        new_img = np.zeros((28, 28), dtype=img_array.dtype)
        y_center = (28 - digit_height) // 2
        x_center = (28 - digit_width) // 2
        new_img[y_center:y_center+digit_height, x_center:x_center+digit_width] = digit
        return new_img
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazka {image_path}: {e}")
        return None

def preprocess_image(image_path):
    try:
        img_array = center_and_pad_image(image_path)
        if img_array is None:
            return None
        img_array = img_array.astype(np.float32) / 255.0
        img_vector = img_array.flatten().reshape(1, -1).astype(np.float32)
        print(f"Zakres pikseli dla {image_path}: min={img_vector.min()}, max={img_vector.max()}")
        return img_vector
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazka {image_path}: {e}")
        return None

def visualize_preprocessed_images(model, image_paths):
    for image_path in image_paths:
        X_custom = preprocess_image(image_path)
        if X_custom is not None:
            original_img = Image.open(image_path).convert('L').resize((28, 28), Image.LANCZOS)
            original_img_array = 255 - np.array(original_img)

            plt.figure(figsize=(4, 2))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img_array, cmap='gray')
            plt.title("Oryginalny")
            plt.axis('off')

            processed_img = X_custom.reshape(28, 28)
            plt.subplot(1, 2, 2)
            plt.imshow(processed_img, cmap='gray')
            plt.title("Przetworzony")
            plt.axis('off')
            plt.show()

def test_custom_image(model, image_path):
    processed_img = center_and_pad_image(image_path)
    if processed_img is not None:
        img_vector = processed_img.flatten().reshape(1, -1).astype(np.float32) / 255.0
        y_pred = model.predict(img_vector)
        pred_label = int(np.argmax(y_pred, axis=1)[0])
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Przewidywana etykieta: {pred_label}")
        plt.axis('off')
        plt.show()

parent_dir = os.path.join(".")

image_paths = glob.glob(os.path.join(parent_dir, "*.png"))

for image_path in image_paths:
    test_custom_image(model, image_path)
