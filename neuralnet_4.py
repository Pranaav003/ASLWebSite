# 4. Training the neural network
# Run this file to train the neural network

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from preprocess_3 import X_train, y_train, actions
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Define your model architecture at the module level
model = Sequential()

# Add an Input layer
model.add(Input(shape=(50, 1662)))

# Add LSTM layers
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Add Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

def main():
    log_dir = os.path.join("Logs")
    tb_callback = TensorBoard(log_dir=log_dir)

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        # Also change
        patience=400,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    # Train the model with validation data and callbacks
    model.fit(
        X_train, y_train,
        validation_split=0.10,
        # Change to change results
        epochs=500,
        callbacks=[tb_callback, early_stopping, checkpoint]
    )

    model.save('final_model.keras')

    # Load the best model saved during training
    model.load_weights('final_model.keras')
    yhat = model.predict(X_train)
    ytrue = np.argmax(y_train, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print("Confusion Matrix: \n", multilabel_confusion_matrix(ytrue, yhat))
    print("Accuracy: ", accuracy_score(ytrue, yhat))

if __name__ == "__main__":
    main()
