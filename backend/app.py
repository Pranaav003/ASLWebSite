import os
import base64
import cv2
import numpy as np
from flask           import Flask, request, jsonify, send_from_directory
from flask_cors      import CORS
from collections     import deque

from init_1          import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
from test_5          import model, actions as MODEL_ACTIONS
from test_gpt        import generate_summary
import test_finger

app = Flask(
    __name__,
    static_folder="frontend/build",
    static_url_path=""
)
CORS(app)

SEQ_LENGTH = 30
SMOOTH_WIN = 10

ASL_CLIENTS = {}  # client_id → {sequence, predictions, sentence, last_probs, holistic}

def get_client_state(cid):
    st = ASL_CLIENTS.get(cid)
    if not st:
        st = {
            "sequence":    deque(maxlen=SEQ_LENGTH),
            "predictions": [],
            "sentence":    [],
            "last_probs":  [],
            "holistic":    mp_holistic.Holistic(
                                static_image_mode=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5
                            )
        }
        ASL_CLIENTS[cid] = st
    return st

@app.route("/actions")
def get_actions():
    return jsonify(actions=MODEL_ACTIONS.tolist())

@app.route("/process_frame", methods=["OPTIONS","POST"])
def process_frame():
    if request.method == "OPTIONS":
        return "", 200

    data      = request.get_json(force=True)
    cid       = data.get("clientId")
    img_data  = data.get("image","")

    if not cid:
        return jsonify(error="Missing clientId"), 400

    state = get_client_state(cid)

    # ── EARLY GUARD: if no valid image string, return last state ──
    if not isinstance(img_data, str) or not img_data.strip():
        return jsonify({
            "probabilities": state["last_probs"],
            "action":        " ".join(state["sentence"])
        }), 200

    # strip data-url prefix
    if "," in img_data:
        _, img_data = img_data.split(",",1)

    try:
        # decode
        raw = base64.b64decode(img_data)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid frame")

        # mediapipe + draw (optional)
        image, results = mediapipe_detection(frame, state["holistic"])
        if results:
            draw_styled_landmarks(image, results)

        # extract & buffer
        kp = extract_keypoints(results) if results else np.zeros(1662)
        state["sequence"].append(kp)

        # predict once buffer full
        probs = np.zeros(len(MODEL_ACTIONS))
        if len(state["sequence"]) == SEQ_LENGTH:
            preds = model.predict(
                np.expand_dims(state["sequence"],axis=0)
            )[0]
            idx = int(np.argmax(preds))
            state["predictions"].append(idx)

            if (len(state["predictions"])>=SMOOTH_WIN
                and state["predictions"][-SMOOTH_WIN:].count(idx)==SMOOTH_WIN
                and preds[idx]>0.8):
                w = MODEL_ACTIONS[idx]
                if not state["sentence"] or state["sentence"][-1]!=w:
                    state["sentence"].append(w)
            probs = preds

        state["last_probs"] = probs.tolist()

        return jsonify({
            "probabilities": probs.tolist(),
            "action":        " ".join(state["sentence"])
        }), 200

    except Exception as e:
        print("❌ /process_frame error:", e)
        # fallback to last good state
        return jsonify({
            "probabilities": state["last_probs"],
            "action":        " ".join(state["sentence"])
        }), 200

@app.route("/process_finger_frame", methods=["OPTIONS","POST"])
def process_finger_frame():
    if request.method=="OPTIONS":
        return "",200

    data     = request.get_json(force=True)
    img_data = data.get("image","")
    if "," in img_data:
        _, img_data = img_data.split(",",1)

    try:
        raw   = base64.b64decode(img_data)
        arr   = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid frame")

        # reset per-call
        test_finger.SENTENCE  = []
        test_finger.LAST_TIME = 0.0

        _, sl = test_finger.run_finger_on_frame(frame)
        char = sl[-1] if sl else ""
        return jsonify({ "action": char }), 200

    except Exception as e:
        print("❌ /process_finger_frame error:", e)
        return jsonify({ "action": "" }), 200

@app.route("/summary", methods=["POST"])
def summary():
    data      = request.get_json(force=True)
    sentences = data.get("sentences",[])
    try:
        txt = generate_summary(sentences)
    except Exception as e:
        print("❌ /summary error:",e)
        txt = ""
    return jsonify(summary=txt),200

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__=="__main__":
    port = int(os.getenv("PORT","5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
