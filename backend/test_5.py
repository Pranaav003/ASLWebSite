import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

model      = load_model("final_model.keras")
actions    = np.array(["hello", "iloveyou", "thanks"])

# Enable static_image_mode to avoid timestamp-mismatch errors
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQ_LENGTH  = 30
SMOOTH_WIN  = 10
sequence    = deque(maxlen=SEQ_LENGTH)
predictions = []
sentence    = []

def run_test_on_frame(frame):
    """
    Returns (annotated_frame, probabilities, sentence_list).
    """
    # Mediapipe detection + drawing
    image, results = mediapipe_detection(frame, holistic)
    if results:
        draw_styled_landmarks(image, results)

    # keypoints -> sequence buffer
    keypoints = extract_keypoints(results) if results else np.zeros(1662)
    sequence.append(keypoints)

    # default zero-probs
    probs = np.zeros(len(actions))
    if len(sequence) == SEQ_LENGTH:
        preds = model.predict(np.expand_dims(sequence, axis=0))[0]
        idx   = int(np.argmax(preds))
        predictions.append(idx)

        # smoothing
        if (len(predictions) >= SMOOTH_WIN and
            predictions[-SMOOTH_WIN:].count(idx) == SMOOTH_WIN and
            preds[idx] > 0.8):
            word = actions[idx]
            if not sentence or sentence[-1] != word:
                sentence.append(word)
        probs = preds

    return image, probs, list(sentence)
