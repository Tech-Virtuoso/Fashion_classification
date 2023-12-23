
---

# Fashion Classification Model using CNN

## Overview:

This project implements a Convolutional Neural Network (CNN) for the classification of fashion items using the Fashion MNIST dataset. The model is built using the Keras Sequential API and achieves an accuracy of 90% after 10 epochs with a batch size of 32.

## Key Features:

- **Dataset:** Utilizes the Fashion MNIST dataset, consisting of grayscale images of 10 different fashion categories.
  
- **Model Architecture:** Sequential model with Conv2D layers, MaxPooling2D layers, Flatten layer, and Dense layers. The activation function used is ReLU for hidden layers, and softmax for the output layer.

- **Training Configuration:** Trained for 10 epochs with a batch size of 32. The Adam optimizer with a learning rate of 0.001 is used, and the model is compiled with categorical crossentropy loss.

- **Achievements:** Achieves a classification accuracy of 90% on the test set.

## How to Use:


1. **Training the Model:**

   ```python
   # Load and preprocess the data (ensure Fashion MNIST is available)
   (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
   X_train = X_train / 255.0
   X_test = X_test / 255.0
   y_train = to_categorical(y_train)
   y_test = to_categorical(y_test)

   # Build and train the model
   model = Sequential(...)
   model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
   ```

2. **Evaluation:**

   ```python
   # Evaluate the model on the test set
   test_loss, test_acc = model.evaluate(X_test, y_test)
   print(f'Test accuracy: {test_acc}')
   ```

## Results:

- **Training History:** Visualize the training and validation accuracy over epochs.

   ```python
   # Visualize training history
   plt.plot(history.history['accuracy'], label='Training Accuracy')
   plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()
   ```

## License:

This project is licensed under the MIT Linces.

Feel free to explore and contribute to the project! If you have any questions or suggestions, please open an issue.

---

