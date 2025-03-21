# streamlit app
# Run only when model fully trained on symbols, and is tested with test_5.py

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time  # To track the timing for ChatGPT calls
from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from test_5 import prob_viz
from preprocess_3 import actions
from tensorflow import keras
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

# Initialize OpenAI API client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the trained model
load_model = keras.models.load_model
try:
    model = load_model('final_model.keras')  # Ensure this path is correct
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None  # Safeguard to prevent further errors if model loading fails

st.title('ASL Gesture Recognition')
st.write('''### Click the **checkbox** below to start the webcam feed. Perform ***American Sign Language*** gestures in front of the camera to see the **model predictions** as ***subtitles*** on the video feed.''')

st.write('The model is trained on the following gestures: ' + ', '.join([f'**{action}**' for action in actions]) + '.')

# Create a checkbox to start and stop the feed
run = st.checkbox('Run')

# Initialize session state
if 'run' not in st.session_state:
    st.session_state['run'] = False
if 'cap' not in st.session_state:
    st.session_state['cap'] = None
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = []
if 'last_chatgpt_time' not in st.session_state:
    st.session_state['last_chatgpt_time'] = time.time()

# Update session state based on checkbox
st.session_state['run'] = run

# Create a placeholder for the video frames
FRAME_WINDOW = st.image([])
# Create a placeholder for ChatGPT predictions
CHATGPT_PLACEHOLDER = st.empty()

# Initialize mediapipe and other variables
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Color setup for probabilities visualization
num_symbols = len(actions)
colormap = plt.cm.get_cmap('hsv', num_symbols)
colors = [colormap(i)[:3] for i in range(num_symbols)]
colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]

def release_capture():
    """Safely release the video capture object."""
    try:
        if st.session_state['cap'] is not None and st.session_state['cap'].isOpened():
            st.session_state['cap'].release()
        st.session_state['cap'] = None
        cv2.destroyAllWindows()
    except Exception:
        # Suppress errors silently during cleanup
        pass

def get_chatgpt_guess(sentence):
    """Send the current sentence to ChatGPT and get guesses."""
    if not sentence or not client:
        return "No valid sentence or ChatGPT client to process."
    try:
        input_text = " ".join(sentence)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that understands the context of ASL gestures."},
            {"role": "user", "content": f"The user signed: '{input_text}'. Provide 2-3 extremely simple possible English sentences based on the context. Try not to use words that were not signed, only filler words."},
        ]
        completion = client.chat.completions.create(model=MODEL, messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error with ChatGPT API: {str(e)}"

def main():
    if model is None:
        st.error("Model not loaded. Please check the configuration.")
        return

    # Initialize video capture only when running
    if st.session_state['cap'] is None:
        st.session_state['cap'] = cv2.VideoCapture(0)

    sequence = []
    predictions = []
    threshold = 0.6  # Updated to match test_5.py

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,  # Updated to match test_5.py
        min_tracking_confidence=0.4
    ) as holistic:
        while st.session_state['run']:
            # Read Feed
            if st.session_state['cap'] is None or not st.session_state['cap'].isOpened():
                st.write("Camera feed not available.")
                break

            ret, frame = st.session_state['cap'].read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Flip the image for a mirror effect
            frame = cv2.flip(frame, 1)

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            if image is None or results is None:
                continue  # Skip this frame if detection failed

            # Draw Landmarks
            draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep last 30 keypoints

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if res[np.argmax(res)] > threshold:
                    if len(st.session_state['sentence']) == 0 or actions[np.argmax(res)] != st.session_state['sentence'][-1]:
                        st.session_state['sentence'].append(actions[np.argmax(res)])

                # Trigger ChatGPT guesses every 20 seconds
                current_time = time.time()
                if current_time - st.session_state['last_chatgpt_time'] >= 20:
                    guesses = get_chatgpt_guess(st.session_state['sentence'])
                    CHATGPT_PLACEHOLDER.markdown(f"**Did you mean:**\n{guesses}")
                    st.session_state['last_chatgpt_time'] = current_time

                # Visualization probabilities
                image = prob_viz(res, actions, image, colors)

            # Add the black rectangle and subtitles
            height, width, _ = image.shape
            rectangle_bgr = (0, 0, 0)  # Black rectangle for subtitles
            cv2.rectangle(image, (0, height - 100), (width, height), rectangle_bgr, -1)
            cv2.putText(
                image,
                ' '.join(st.session_state['sentence']),  # Display all recognized symbols
                (10, height - 10),  # Position text slightly above the bottom
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # Font scale
                (255, 255, 255),  # White text
                2,  # Thickness
                cv2.LINE_AA
            )

            # Convert BGR image to RGB for Streamlit
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image in the Streamlit app
            FRAME_WINDOW.image(image)

        # Release the capture when stopped
        release_capture()

# Start the main process if running
if st.session_state['run']:
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred during the live feed: {e}")
else:
    release_capture()
    st.write('Webcam feed stopped. Toggle "Run" to restart.')

st.write('---')
st.write('Created using **Streamlit, OpenCV, Mediapipe, and Tensorflow/Keras.**')
st.write('This project was made to demonstrate the use of AI in recognizing American Sign Language gestures, and to showcase the integration of AI models with web applications.')
st.write('The hope is to increase accessibility and inclusivity for the hearing-impaired community.')
st.write('Made with ❤️, Pranaav Iyer')
