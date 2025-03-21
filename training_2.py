# 2. Collecting Training Data
# Run this file to collect data for training the model

import cv2
import numpy as np
import os
from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp
mp_holistic = mp.solutions.holistic
from init_1 import mp_holistic, actions, DATA_PATH

# 30 videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

def main():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cv2.startWindowThread()

    # If no popup, try 1 or 2
    cap = cv2.VideoCapture(0)

    # Access mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read Feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw Landmarks
                    draw_styled_landmarks(image, results)

                    # Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, "STARTING COLLECTION", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f"Collecting frames for {action} Video Number {sequence}", (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)
                        cv2.waitKey(1000)
                    else:
                        cv2.putText(image, f"Collecting frames for {action} Video Number {sequence}", (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)

                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
