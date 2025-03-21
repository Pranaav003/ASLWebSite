# 3. Preprocess the Data
# Run this file to preprocess the data for training the model


import numpy as np
import os
from training_2 import actions, no_sequences, sequence_length, DATA_PATH
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [],[]
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

y = to_categorical(labels).astype(int)

def trim_frame(frame, expected_length=1662):
    if frame.shape[0] > expected_length:
        # Trim the frame to the expected length
        return frame[:expected_length]
    elif frame.shape[0] < expected_length:
        # Pad the frame with zeros if it's shorter
        padding = np.zeros(expected_length - frame.shape[0])
        return np.concatenate([frame, padding])
    else:
        return frame

for i, seq in enumerate(sequences):
    fixed_seq = []
    for j, frame in enumerate(seq):
        fixed_frame = trim_frame(np.array(frame))
        fixed_seq.append(fixed_frame)
    sequences[i] = fixed_seq

X = np.array(sequences)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

print("Preprocessing Successful")