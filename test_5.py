import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess_3 import X_test, actions
from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from collections import deque

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic

# Load trained model
model = load_model('final_model.keras')

# Ensure colors are distinct for each action
num_symbols = len(actions)
def distinct_colors(n):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    additional_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Yellow, Magenta, Cyan
    colors = base_colors + additional_colors[:max(0, n - len(base_colors))]
    return colors[:n]

colors = distinct_colors(num_symbols)

def prob_viz(res, actions, input_frame, colors):
    """Visualizes prediction probabilities on the frame."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if num >= len(colors):
            continue
        cv2.rectangle(output_frame, (0, 60 + num * 40),
                      (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob:.2f}', (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def run_test_model():
    sequence, sentence, predictions = [], [], []
    threshold = 0.8
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.4) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_index = np.argmax(res)
                predictions.append(predicted_index)

                if len(predictions) >= 10:
                    recent_predictions = predictions[-10:]
                    unique_predictions = np.unique(recent_predictions)
                    if len(unique_predictions) == 1 and unique_predictions[0] == predicted_index:
                        if res[predicted_index] > threshold:
                            if len(sentence) == 0 or actions[predicted_index] != sentence[-1]:
                                sentence.append(actions[predicted_index])
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                
                image = prob_viz(res, actions, image, colors)

            height, width, _ = image.shape
            cv2.rectangle(image, (0, height - 100), (width, height), (0, 0, 0), -1)
            cv2.putText(image, ' '.join(sentence), (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test_model()
