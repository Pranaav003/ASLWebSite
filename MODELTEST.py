import cv2
import mediapipe as mp
import numpy as np
from keras import models
import os
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.preprocessing import LabelEncoder
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

current_dir = os.path.dirname(__file__)
modelFile = "asl_lstm_model2.h5"
yEncoderFile = "label_encoder2.pkl"


model = models.load_model(os.path.join(current_dir, modelFile))
label_encoder = joblib.load(os.path.join(current_dir, yEncoderFile))


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
  while cap.isOpened():
    success, image = cap.read()
    h, w, c = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    landmarksArray = []

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      result_handedness = results.multi_handedness[0].classification[0].label

      # Find the bounds of the hand_landmarks (in pixels).
      # Then, normalize the pixel amounts to between [0, 1].
      image_height, image_width, _ = image.shape
      for hand_landmarks in results.multi_hand_landmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        
        for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
        for lm in hand_landmarks.landmark:
          cx = ((lm.x*image_width)-x_min)/(x_max-x_min)
          cy = ((lm.y*image_height)-y_min)/(y_max-y_min)
          
          # The model is currently trained on only one hand,
          # so if the handedness is left, we flip the x coordinates of the landmarks
          # before we give that landmark data to the model.
          if (result_handedness == "Left"):
            cx = abs(1-cx)
          
          # This is the array that will be used as input to the model.
          landmarksArray.append([cx, cy, lm.z])

        # Display a rectangle around the bounds, padding optional
        rectanglePadding = 0
        cv2.rectangle(image, (x_min-rectanglePadding, y_min-rectanglePadding), (x_max+rectanglePadding, y_max+rectanglePadding), (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        #print("Hand in frame. x-max: "+str(x_max)+" x-min: "+str(x_min)+" y-max: "+str(y_max)+" y-min: "+str(y_min))

        # Reshape the data so the model can read it
        input_data = np.array(landmarksArray).reshape(1, 21, 3)  # Shape: (1, 21, 3)
        landmarksArray.clear()

        # Make a prediction
        predictions = model.predict(input_data, verbose=0)

        # Get the index of the highest probability
        predicted_index = np.argmax(predictions, axis=1)[0]  # Get the predicted class index

        # Decode the prediction to get the character
        predicted_character = label_encoder.inverse_transform([predicted_index])[0]

        confidence = predictions[0][predicted_index]  # Probability of the predicted character

        print(f"Predicted Character: {predicted_character} Handedness: {result_handedness}")
        flipped_image = cv2.flip(image, 1)
        cv2.putText(flipped_image, predicted_character+":"+str(float(f"{confidence:.3f}") ), (abs((x_max+10)-w), y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    else: 
      print("No hand in frame.")
      flipped_image = cv2.flip(image, 1)

    cv2.imshow('MediaPipe Hands', flipped_image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()