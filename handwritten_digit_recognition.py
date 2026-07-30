# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# Verify TensorFlow version
print("TensorFlow Version:", tf.__version__)


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0


# Reshape data to fit CNN input
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))


# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Display model architecture
model.summary()


# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc:.4f}")


# Save trained model
model.save("digit_recognition_model.h5")


# Plot training results
plt.figure(figsize=(12, 4))


# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],
         label='Training Accuracy')
plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],
         label='Training Loss')
plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# Display predictions
def display_predictions(model, x_test, y_test):

    predictions = model.predict(x_test)

    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)

        plt.imshow(
            x_test[i].reshape(28, 28),
            cmap='gray'
        )

        plt.title(
            f"Prediction: {np.argmax(predictions[i])} | Actual: {y_test[i]}"
        )

        plt.axis("off")

    plt.show()


display_predictions(model, x_test, y_test)