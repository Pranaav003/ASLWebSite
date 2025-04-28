# backend/app.py

import os
import time
import cv2
import numpy as np
import openai
import joblib

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
import mediapipe as mp
from tensorflow.keras.models import load_model

# ─── helper imports ─────────────────────────────────────────────────────────────
from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
from test_5 import run_test_on_frame, actions as asl_actions

# ─── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="frontend/build",   # ← React build output
    static_url_path="/"               # ← serve at root
)
CORS(app)

# ─── Load environment & keys ────────────────────────────────────────────────────
load_dotenv()  
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── Camera index from ENV (default 0) ─────────────────────────────────────────
CAM_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

latest_summary = ""


# ─── ASL MJPEG stream + ChatGPT summarization ─────────────────────────────────
def gen_frames_asl():
    global latest_summary
    cap = cv2.VideoCapture(CAM_INDEX)          # ← use CAM_INDEX
    last_chat = time.time()

    COLORS = [(255,0,0),(0,255,0),(0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1) inference + draw
        annotated, probs, sentence = run_test_on_frame(frame)
        h, w, _ = annotated.shape

        # 2) draw bars
        bx, by, bh, sp = 10, 10, 25, 15
        maxw = int(w * 0.15)
        fs, th = 1.2, 2
        for i,(act,p) in enumerate(zip(asl_actions, probs)):
            y0 = by + i*(bh+sp)
            lw = int(p * maxw)
            cv2.rectangle(annotated,(bx,y0),(bx+maxw,y0+bh),(50,50,50),-1)
            cv2.rectangle(annotated,(bx,y0),(bx+lw,y0+bh),COLORS[i],-1)
            cv2.putText(annotated,
                        f"{act} {int(p*100)}%",
                        (bx+maxw+10, y0+bh-5),
                        font, fs, COLORS[i], th)

        # 3) ChatGPT every 10s
        now = time.time()
        if sentence and now - last_chat > 10:
            last_chat = now
            txt = " ".join(sentence)
            msgs = [
                {"role":"system","content":"You are a helpful assistant that understands ASL context."},
                {"role":"user",
                 "content":(
                   f"The user signed: '{txt}'. "
                   "Provide 2–3 extremely simple English sentences based on that."
                 )}
            ]
            try:
                resp = openai.chat.completions.create(
                    model="gpt-3.5-turbo", messages=msgs, max_tokens=50
                )
                latest_summary = resp.choices[0].message.content.strip()
            except Exception as e:
                print("ChatGPT error:", e)

        # 4) sentence bar
        barh = 80
        cv2.rectangle(annotated, (0, h-barh), (w, h), (0,0,0), -1)
        cv2.putText(annotated,
                    " ".join(sentence),
                    (10, h-20),
                    font, 2.0, (255,255,255), 4)

        # 5) yield MJPEG chunk
        ret, buf = cv2.imencode('.jpg', annotated)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )

    cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames_asl(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─── Finger‐sign MJPEG stream ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
finger_model  = load_model(os.path.join(BASE, "asl_lstm_model2.h5"))
label_encoder = joblib.load(os.path.join(BASE, "label_encoder2.pkl"))

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def gen_frames_fingersign():
    cap = cv2.VideoCapture(CAM_INDEX)        # ← use CAM_INDEX
    hands = mp_hands.Hands(
       model_complexity=0,
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5,
       max_num_hands=1
    )

    sentence = []
    last_recog = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            # bounding box
            xs, ys = [], []
            for lm in results.multi_hand_landmarks[0].landmark:
                x,y = int(lm.x*w), int(lm.y*h)
                xs.append(x); ys.append(y)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # normalize + predict
            arr = []
            for lm in results.multi_hand_landmarks[0].landmark:
                cx = ((lm.x*w)-x_min)/max(x_max-x_min,1)
                cy = ((lm.y*h)-y_min)/max(y_max-y_min,1)
                arr.append([cx,cy,lm.z])
            inp = np.array(arr).reshape(1,21,3)
            preds = finger_model.predict(inp, verbose=0)[0]
            idx  = int(np.argmax(preds))
            char = label_encoder.inverse_transform([idx])[0]
            conf = preds[idx]

            now = time.time()
            if conf >= 0.90:
                if not sentence or sentence[-1] != char:
                    sentence.append(char)
                    last_recog = now
                elif now - last_recog >= 1.5:
                    sentence.append(char)
                    last_recog = now

            # draw box + landmarks + overlay
            cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
            mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            cv2.putText(frame,
                        f"{char}:{conf:.2f}",
                        (x_min, y_min-20),
                        font, 2.0, (0,0,0), 2)

        # sentence bar
        barh = 100
        cv2.rectangle(frame, (0, h-barh), (w, h), (0,0,0), -1)
        cv2.putText(frame,
                    "".join(sentence),
                    (10, h-20),
                    font, 2.5, (255,255,255), 5, cv2.LINE_AA)

        ret, buf = cv2.imencode('.jpg', frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )

    hands.close()
    cap.release()


@app.route("/finger_feed")
def finger_feed():
    return Response(
        gen_frames_fingersign(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─── summary + actions endpoints ────────────────────────────────────────────────
@app.route("/summary")
def summary():
    return jsonify({"summary": latest_summary})

@app.route("/actions")
def actions_list():
    return jsonify({"actions": asl_actions.tolist()})


# ─── catch-all to serve React SPA ─────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    static_dir = app.static_folder
    full = os.path.join(static_dir, path)
    if path and os.path.exists(full):
        return send_from_directory(static_dir, path)
    return send_from_directory(static_dir, "index.html")


# ─── run! ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))    # default to 5001
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
