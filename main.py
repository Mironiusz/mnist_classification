import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.datasets import mnist

from neural_network import NeuralNetwork, cross_entropy_loss, accuracy


# --- Przygotowanie danych MNIST ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizacja i spłaszczenie
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32)  / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# One-hot encoding
def to_one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    for i, val in enumerate(y):
        oh[i, val] = 1.0
    return oh

y_train_oh = to_one_hot(y_train)
y_test_oh  = to_one_hot(y_test)

# --- Inicjalizacja i trenowanie sieci ---
model = NeuralNetwork(lr=0.01)
batch_size = 128
epochs = 5  # możesz zwiększyć w razie potrzeby
num_train = x_train.shape[0]
num_batches = num_train // batch_size

for e in range(epochs):
    # Mieszamy indeksy, żeby dane były wybierane losowo
    idx = np.arange(num_train)
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

# --- Test ---
y_pred_test = model.predict(x_test)
test_loss = cross_entropy_loss(y_pred_test, y_test_oh)
test_acc  = accuracy(y_pred_test, y_test_oh)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")





# --- Funkcje do wczytywania własnych obrazów ---

def center_and_pad_image(image_path):
    """
    Wczytuje obrazek, zmienia jego rozmiar do 28x28, centruje cyfrę
    i dodaje odpowiedni margines.
    """
    try:
        # Wczytaj obrazek i konwertuj do skali szarości
        img = Image.open(image_path).convert('L')

        # Zmień rozmiar z zachowaniem proporcji
        img.thumbnail((20, 20), Image.LANCZOS)

        # Przekształć obrazek na tablicę NumPy
        img_array = np.array(img)

        # Inwersja kolorów (czarne cyfry na białym tle)
        img_array = 255 - img_array

        # Oblicz marginesy
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        if not rows.any() or not cols.any():
            return img_array  # Pusty obrazek

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Wytnij cyfrę
        digit = img_array[y_min:y_max+1, x_min:x_max+1]

        # Oblicz nowe położenie cyfry
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
    """
    Wczytuje plik 'image_path' i przygotowuje go w formacie 28x28 (skala szarości),
    tak jak dane MNIST.
    """
    try:
        img_array = center_and_pad_image(image_path)
        if img_array is None:
            return None
        # Normalizacja
        img_array = img_array.astype(np.float32) / 255.0
        # Spłaszczanie do wektora 1x784
        img_vector = img_array.flatten().reshape(1, -1).astype(np.float32)

        # Debugging: Print min and max
        print(f"Zakres pikseli dla {image_path}: min={img_vector.min()}, max={img_vector.max()}")

        return img_vector
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazka {image_path}: {e}")
        return None

def visualize_preprocessed_images(model, image_paths):
    for image_path in image_paths:
        X_custom = preprocess_image(image_path)
        if X_custom is not None:
            # Wyświetlenie oryginalnego obrazka
            original_img = Image.open(image_path).convert('L').resize((28, 28), Image.LANCZOS)
            original_img_array = 255 - np.array(original_img)

            plt.figure(figsize=(4, 2))

            plt.subplot(1, 2, 1)
            plt.imshow(original_img_array, cmap='gray')
            plt.title("Oryginalny")
            plt.axis('off')

            # Wyświetlenie przetworzonego obrazka
            processed_img = X_custom.reshape(28, 28)
            plt.subplot(1, 2, 2)
            plt.imshow(processed_img, cmap='gray')
            plt.title("Przetworzony")
            plt.axis('off')

            plt.show()

# --- Wizualizacja przetworzonych obrazków ---
image_paths = ["zero.png", "jeden.png", "dwa.png", "trzy.png",
               "cztery.png", "piec.png", "szesc.png", "siedem.png",
               "osiem.png", "dziewiec.png"]

#image_paths = ["zero.png", "jeden.png"]

# visualize_preprocessed_images(model, image_paths)


def test_custom_image(model, image_path):
    """
    Ładuje obraz z dysku, przetwarza i pokazuje wynik sieci.
    """
    processed_img = center_and_pad_image(image_path)
    if processed_img is not None:
        img_vector = processed_img.flatten().reshape(1, -1).astype(np.float32) / 255.0
        y_pred = model.predict(img_vector)
        pred_label = int(np.argmax(y_pred, axis=1)[0])

        # Wyświetlenie obrazka z przewidywaną etykietą
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Przewidywana etykieta: {pred_label}")
        plt.axis('off')
        plt.show()


# --- Przykładowe testowanie własnych obrazów ---
for image_path in image_paths:
    test_custom_image(model, image_path)
