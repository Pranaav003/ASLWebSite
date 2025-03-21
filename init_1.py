#1. Detection Variables
# Run this file to test the mediapipe library and the webcam feed
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
import cv2
import numpy as np
import mediapipe as mp
import os


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    if image is None:
        print("Empty image received in mediapipe_detection")
        return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False                   # Image is no longer writable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    if results:
        # Draw Face Landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
        # Draw Pose Landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        # Draw Left Hand Landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        # Draw Right Hand Landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

def extract_keypoints(results):
    # Initialize empty arrays for landmarks with zeros
    pose = np.zeros(132)     # 33 landmarks * 4 (x, y, z, visibility)
    face = np.zeros(1404)    # 468 landmarks * 3 (x, y, z)
    lh = np.zeros(63)        # 21 landmarks * 3 (x, y, z)
    rh = np.zeros(63)        # 21 landmarks * 3 (x, y, z)

    # Extract pose landmarks if detected
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]).flatten()

    # Extract face landmarks if detected
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        face = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks]).flatten()

    # Extract left hand landmarks if detected
    if results.left_hand_landmarks:
        lh_landmarks = results.left_hand_landmarks.landmark
        lh = np.array([[lm.x, lm.y, lm.z] for lm in lh_landmarks]).flatten()

    # Extract right hand landmarks if detected
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
        rh = np.array([[lm.x, lm.y, lm.z] for lm in rh_landmarks]).flatten()

    # Concatenate all arrays
    keypoints = np.concatenate([pose, face, lh, rh])
    return keypoints

def main():
    cv2.startWindowThread()

    # If no popup, try 1 or 2
    cap = cv2.VideoCapture(0)

    # Access Mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read Feed
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            if image is None or results is None:
                continue  # Skip this frame if detection failed

            # Draw Landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # Example of how to process the results after the loop
    if results:
        keypoints = extract_keypoints(results)
        # Do something with keypoints

if __name__ == "__main__":
    main()

# Actions to try and detect
actions = np.array(["presentation","project","translate"])

# Path for exported data, np arrays
DATA_PATH = os.path.join("MP_Data")