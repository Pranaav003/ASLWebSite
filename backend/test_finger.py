# backend/test_finger.py

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib

# Load your LSTM model & label encoder
BASE = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE, "asl_lstm_model2.h5")
ENCODER_PATH = os.path.join(BASE, "label_encoder2.pkl")

model         = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Mediapipe hands in static mode
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# State for smoothing & sentence
LAST_TIME = 0.0
SENTENCE  = []

def run_finger_on_frame(frame):
    """
    Returns annotated frame and current SENTENCE list of characters,
    repeating the same character if held for >=1.5s.
    """
    global LAST_TIME, SENTENCE

    annotated = frame.copy()
    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb)
    h, w, _   = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # bounding box
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = min(xs)*w, max(xs)*w
        y_min, y_max = min(ys)*h, max(ys)*h

        # normalize landmarks into [21,3]
        coords = []
        for lm in hand_landmarks.landmark:
            cx = (lm.x * w - x_min) / (x_max - x_min + 1e-6)
            cy = (lm.y * h - y_min) / (y_max - y_min + 1e-6)
            coords.append([cx, cy, lm.z])
        input_data = np.array(coords).reshape(1, 21, 3)

        # predict
        preds = model.predict(input_data, verbose=0)[0]
        idx   = int(np.argmax(preds))
        char  = label_encoder.inverse_transform([idx])[0]
        conf  = preds[idx]

        now = time.time()
        if conf >= 0.90:
            # if new char, always append
            if not SENTENCE or SENTENCE[-1] != char:
                SENTENCE.append(char)
                LAST_TIME = now
            # if same char held >1.5s, append again
            elif now - LAST_TIME >= 1.5:
                SENTENCE.append(char)
                LAST_TIME = now

        # draw landmarks & box
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )
        cv2.rectangle(
            annotated,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            (0,255,0), 2
        )
        cv2.putText(
            annotated,
            f"{char}:{conf:.2f}",
            (int(x_min), int(y_min)-10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2
        )

    # draw bottom sentence bar
    cv2.rectangle(annotated, (0, h-60), (w, h), (0,0,0), -1)
    cv2.putText(
        annotated,
        "".join(SENTENCE),
        (10, h-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        2, (255,255,255), 3, cv2.LINE_AA
    )

    return annotated, SENTENCE
