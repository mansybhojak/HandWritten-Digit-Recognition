# -*- coding: utf-8 -*-
"""ml.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vZPVFI6qm5KKqx-fA1nOcKObJHnMt04W
"""

!pip install tensorflow matplotlib

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize (scale) the image data from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Convert labels to one-hot encoded format
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)     # (60000, 28, 28)
print(y_train_cat.shape) # (60000, 10)

import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28)),        # Flatten 28x28 to 784
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')       # 10 classes for digits 0-9
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print("Test Accuracy:", test_acc)

# Predict
predictions = model.predict(x_test)

# Show first 5 predictions
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {predictions[i].argmax()} | Actual: {y_test[i]}")
    plt.show()