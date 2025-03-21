import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from keras import models
import os
import joblib
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.preprocessing import LabelEncoder

# Initialize Streamlit app
st.title('ASL Fingersigning Model')
st.write('''
### Click the **checkbox** below to start the webcam feed.
Perform ***American Sign Language*** fingersigning in front of the camera to see the **model predictions** as ***subtitles*** on the video feed.
''')

# Checkbox to control the webcam feed
run = st.checkbox('Run')

# Load model and encoder
current_dir = os.path.dirname(__file__)
modelFile = "asl_lstm_model2.h5"
yEncoderFile = "label_encoder2.pkl"

try:
    model = models.load_model(os.path.join(current_dir, modelFile))
    label_encoder = joblib.load(os.path.join(current_dir, yEncoderFile))
except Exception as e:
    st.error(f"Error loading model or encoder: {e}")
    st.stop()

# Initialize Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Display area for the webcam feed
FRAME_WINDOW = st.image([])

# Sentence display bar (with delay control)
if 'sentence_fingersigning' not in st.session_state:
    st.session_state['sentence_fingersigning'] = []
if 'last_recognition_time' not in st.session_state:
    st.session_state['last_recognition_time'] = 0

# Ensure webcam capture is initialized in session state
if 'cap_fingersigning' not in st.session_state:
    st.session_state['cap_fingersigning'] = None

# Recognition Logic
if run:
    if st.session_state['cap_fingersigning'] is None:
        st.session_state['cap_fingersigning'] = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:

        while run:
            success, image = st.session_state['cap_fingersigning'].read()
            if not success:
                continue

            h, w, c = image.shape
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            landmarksArray = []

            # Draw hand annotations
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                result_handedness = results.multi_handedness[0].classification[0].label
                image_height, image_width, _ = image.shape

                for hand_landmarks in results.multi_hand_landmarks:
                    x_max, y_max, x_min, y_min = 0, 0, w, h

                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_max, x_min = max(x, x_max), min(x, x_min)
                        y_max, y_min = max(y, y_max), min(y, y_min)

                    for lm in hand_landmarks.landmark:
                        cx = ((lm.x * image_width) - x_min) / (x_max - x_min)
                        cy = ((lm.y * image_height) - y_min) / (y_max - y_min)
                        if result_handedness == "Left":
                            cx = abs(1 - cx)
                        landmarksArray.append([cx, cy, lm.z])

                    # Draw rectangle for visual boundary
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Reshape and predict
                    input_data = np.array(landmarksArray).reshape(1, 21, 3)
                    landmarksArray.clear()

                    predictions = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(predictions, axis=1)[0]
                    predicted_character = label_encoder.inverse_transform([predicted_index])[0]
                    confidence = predictions[0][predicted_index]

                    # 1.5 second delay for recognition
                    current_time = time.time()
                    if confidence >= 0.90 and current_time - st.session_state['last_recognition_time'] >= 1.5:
                        st.session_state['sentence_fingersigning'].append(predicted_character)
                        st.session_state['last_recognition_time'] = current_time

                    # Display prediction on screen
                    flipped_image = cv2.flip(image, 1)
                    cv2.putText(flipped_image,
                                f"{predicted_character}: {confidence:.3f}",
                                (abs((x_max + 10) - w), y_min - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 0, 0), 2)

            else:
                flipped_image = cv2.flip(image, 1)

            # Enhanced Sentence Bar
            cv2.rectangle(flipped_image, (0, h - 100), (w, h), (0, 0, 0), -1)
            cv2.putText(flipped_image, ''.join(st.session_state['sentence_fingersigning']),
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5,  # Bigger font size for better visibility
                        (255, 255, 255), 5, cv2.LINE_AA)

            FRAME_WINDOW.image(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))

        st.session_state['cap_fingersigning'].release()
        st.session_state['cap_fingersigning'] = None

else:
    if 'cap_fingersigning' in st.session_state and st.session_state['cap_fingersigning'] is not None:
        st.session_state['cap_fingersigning'].release()
        st.session_state['cap_fingersigning'] = None
    cv2.destroyAllWindows()

# Footer and Project Info
st.write('---')
st.write('Created using **Streamlit, OpenCV, Mediapipe, and Tensorflow/Keras.**')
st.write('This project was made to demonstrate the use of AI in recognizing American Sign Language gestures, and to showcase the integration of AI models with web applications.')
st.write('The hope is to increase accessibility and inclusivity for the hearing-impaired community.')
st.write('Made with ❤️, Pranaav Iyer')
