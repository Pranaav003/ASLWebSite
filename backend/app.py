# backend/app.py

import os
import time
import cv2
import numpy as np
import openai
import joblib
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, flash
from flask_cors import CORS
import mediapipe as mp
from tensorflow.keras.models import load_model

from test_5 import run_test_on_frame, actions as asl_actions

# ─── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "supersecret-key"
CORS(app)  # allow all origins

# ─── OpenAI setup ────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

latest_summary = ""

# ─── ASL MJPEG stream w/ jump‐to‐tail sentence bar ────────────────────────────
def gen_frames_asl():
    global latest_summary

    cap = cv2.VideoCapture(1)  # adjust camera index if needed
    last_chat_time = time.time()

    font   = cv2.FONT_HERSHEY_SIMPLEX
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1) inference & landmark drawing
        annotated, probs, sentence = run_test_on_frame(frame)
        h, w, _ = annotated.shape

        # 2) draw probability bars
        bx, by, bh, sp = 10, 10, 25, 15
        maxw = int(w * 0.15)
        fs, th = 1.2, 2
        for i, (act, p) in enumerate(zip(asl_actions, probs)):
            y0 = by + i * (bh + sp)
            ln = int(p * maxw)
            cv2.rectangle(annotated, (bx, y0), (bx + maxw, y0 + bh), (50,50,50), -1)
            cv2.rectangle(annotated, (bx, y0), (bx + ln, y0 + bh), COLORS[i], -1)
            cv2.putText(
                annotated,
                f"{act} {int(p*100)}%",
                (bx + maxw + 10, y0 + bh - 5),
                font, fs, COLORS[i], th
            )

        # 3) every 10s, update ChatGPT summary
        now = time.time()
        if now - last_chat_time > 10 and sentence:
            last_chat_time = now
            txt = " ".join(sentence)
            messages = [
                {"role":"system",
                 "content":"You are a helpful assistant that understands the context of ASL gestures."},
                {"role":"user",
                 "content":(
                     f"The user signed: '{txt}'. "
                     "Provide 2-3 extremely simple possible English sentences based on the context. "
                     "Try not to use words that were not signed, only filler words."
                 )}
            ]
            try:
                resp = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=50
                )
                raw = resp.choices[0].message.content.strip()
                clean = []
                for line in raw.split("\n"):
                    low = line.lower()
                    if "the user signed" in low or "the user greeted with a sign for" in low:
                        continue
                    clean.append(line.strip())
                latest_summary = "\n".join(clean)
            except Exception as e:
                print("ChatGPT error:", e)

        # 4) draw jump‐to‐tail sentence bar at bottom
        sent_text = " ".join(sentence) if sentence else ""
        barh = 80
        cv2.rectangle(annotated, (0, h - barh), (w, h), (0,0,0), -1)

        # compute text width
        (text_w, _), _ = cv2.getTextSize(sent_text, font, 2.0, 4)
        display_w = w - 20  # 10px padding each side

        # if text fits, left‐align; else shift so tail is visible
        if text_w <= display_w:
            x = 10
        else:
            x = 10 + (display_w - text_w)

        cv2.putText(
            annotated,
            sent_text,
            (x, h - 20),
            font, 2.0, (255,255,255), 4
        )

        # 5) stream the frame
        ret, buf = cv2.imencode('.jpg', annotated)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames_asl(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/summary')
def summary():
    return jsonify({"summary": latest_summary})


# ─── Fingersigning MJPEG stream w/ jump‐to‐tail sentence bar ────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
finger_model   = load_model(os.path.join(BASE_DIR, "asl_lstm_model2.h5"))
label_encoder  = joblib.load(os.path.join(BASE_DIR, "label_encoder2.pkl"))

mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

def gen_frames_fingersign():
    cap = cv2.VideoCapture(1)
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    )

    sentence   = []
    last_recog = 0
    font       = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            # compute bounding box
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in results.multi_hand_landmarks[0].landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

            # normalize landmarks
            lm_array = []
            for lm in results.multi_hand_landmarks[0].landmark:
                cx = (lm.x * w - x_min) / max(x_max - x_min, 1)
                cy = (lm.y * h - y_min) / max(y_max - y_min, 1)
                lm_array.append([cx, cy, lm.z])

            # draw box & landmarks
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # predict finger-sign
            inp   = np.array(lm_array).reshape(1, 21, 3)
            preds = finger_model.predict(inp, verbose=0)[0]
            idx   = int(np.argmax(preds))
            char  = label_encoder.inverse_transform([idx])[0]
            conf  = preds[idx]

            now = time.time()
            if conf >= 0.90:
                if not sentence or sentence[-1] != char:
                    sentence.append(char)
                    last_recog = now
                elif now - last_recog >= 1.5:
                    sentence.append(char)
                    last_recog = now

            # overlay prediction
            cv2.putText(
                frame,
                f"{char}:{conf:.2f}",
                (max(x_min-10,0), max(y_min-30,0)),
                font, 2.0, (0,0,0), 2
            )

        # sentence bar (jump‐to‐tail)
        bar_h = 100
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (0,0,0), -1)
        sent_text = "".join(sentence)
        (text_w, _), _ = cv2.getTextSize(sent_text, font, 2.5, 5)
        display_w = w - 20

        if text_w <= display_w:
            x = 10
        else:
            x = 10 + (display_w - text_w)

        cv2.putText(
            frame,
            sent_text,
            (x, h - 20),
            font, 2.5, (255,255,255), 5, cv2.LINE_AA
        )

        # stream
        ret, buf = cv2.imencode('.jpg', frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )

    hands.close()
    cap.release()


@app.route('/finger_feed')
def finger_feed():
    return Response(
        gen_frames_fingersign(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
