import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

model = keras.Sequential({
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
})

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Treino

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {test_acc:.2%}")

predictions = model.predict(x_test[:5])

print("Classes previstas:", np.argmax(predictions, axis=1))
print("Classes reais:", y_test[:5])




