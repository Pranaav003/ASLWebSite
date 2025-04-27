# backend/test_5.py

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

# Load your trained Keras model
model = load_model("final_model.keras")

# The gesture classes
actions = np.array(["hello", "iloveyou", "thanks"])

# Mediapipe Holistic instance
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffers for smoothing
SEQ_LENGTH = 30
SMOOTH_WINDOW = 10
sequence = deque(maxlen=SEQ_LENGTH)
predictions = []
sentence = []

def run_test_on_frame(frame):
    """
    Process a single BGR frame:
    - Detect & draw landmarks
    - Append keypoints to 30-frame buffer
    - Once full, predict and smooth over last SMOOTH_WINDOW
    - Build up an ever-growing sentence list (no 5-word cap)
    Returns (annotated_image, probs_array, sentence_list)
    """
    # 1) Mediapipe detection + drawing
    image, results = mediapipe_detection(frame, holistic)
    if results:
        draw_styled_landmarks(image, results)

    # 2) Extract keypoints & append
    keypoints = extract_keypoints(results) if results else np.zeros(1662)
    sequence.append(keypoints)

    # 3) Predict once buffer is full
    probs = np.zeros(len(actions))
    if len(sequence) == SEQ_LENGTH:
        probs = model.predict(np.expand_dims(sequence, axis=0))[0]
        idx = int(np.argmax(probs))
        predictions.append(idx)

        # 4) Smoothing: require SMOOTH_WINDOW identical preds & threshold
        if (
            len(predictions) >= SMOOTH_WINDOW
            and predictions[-SMOOTH_WINDOW:].count(idx) == SMOOTH_WINDOW
            and probs[idx] > 0.8
        ):
            word = actions[idx]
            if not sentence or sentence[-1] != word:
                sentence.append(word)
            # **Removed** the trimming to last 5 words

    return image, probs, list(sentence)
