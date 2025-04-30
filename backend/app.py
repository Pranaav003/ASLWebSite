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

# ASL buffering parameters
SEQ_LENGTH = 30
SMOOTH_WIN = 10

# Per-client in-memory state
ASL_CLIENTS = {}  # client_id → { sequence, predictions, sentence, last_probs, holistic }

def get_client_state(client_id):
    state = ASL_CLIENTS.get(client_id)
    if not state:
        state = {
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
        ASL_CLIENTS[client_id] = state
    return state

@app.route("/actions", methods=["GET"])
def get_actions():
    return jsonify(actions=MODEL_ACTIONS.tolist())

@app.route("/process_frame", methods=["OPTIONS", "POST"])
def process_frame():
    if request.method == "OPTIONS":
        return "", 200

    data     = request.get_json(force=True)
    client_id = data.get("clientId")
    img_data  = data.get("image", "")

    if not client_id:
        return jsonify(error="Missing clientId"), 400

    state = get_client_state(client_id)

    # — EARLY GUARD: if no image payload, re-return last known state —
    if not isinstance(img_data, str) or not img_data.strip():
        return jsonify({
            "probabilities": state["last_probs"],
            "action":        " ".join(state["sentence"])
        }), 200

    # Strip the base64 header if present
    if "," in img_data:
        _, img_data = img_data.split(",", 1)

    try:
        # Decode the image
        img_bytes = base64.b64decode(img_data)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode frame")

        # Mediapipe detection & drawing
        image, results = mediapipe_detection(frame, state["holistic"])
        if results:
            draw_styled_landmarks(image, results)

        # Extract keypoints & update sequence
        keypoints = extract_keypoints(results) if results else np.zeros(1662)
        state["sequence"].append(keypoints)

        # Default empty probabilities
        probs = np.zeros(len(MODEL_ACTIONS))
        if len(state["sequence"]) == SEQ_LENGTH:
            preds = model.predict(
                np.expand_dims(state["sequence"], axis=0)
            )[0]
            idx = int(np.argmax(preds))
            state["predictions"].append(idx)

            # Smoothing + threshold logic
            if (
                len(state["predictions"]) >= SMOOTH_WIN
                and state["predictions"][-SMOOTH_WIN:].count(idx) == SMOOTH_WIN
                and preds[idx] > 0.8
            ):
                word = MODEL_ACTIONS[idx]
                if not state["sentence"] or state["sentence"][-1] != word:
                    state["sentence"].append(word)
            probs = preds

        # Save for fallback
        state["last_probs"] = probs.tolist()

        return jsonify({
            "probabilities": probs.tolist(),
            "action":        " ".join(state["sentence"])
        }), 200

    except Exception as e:
        # On error, do not clear old sentence – just return last known
        print("❌ /process_frame error:", e)
        return jsonify({
            "probabilities": state["last_probs"],
            "action":        " ".join(state["sentence"])
        }), 200

@app.route("/process_finger_frame", methods=["OPTIONS", "POST"])
def process_finger_frame():
    if request.method == "OPTIONS":
        return "", 200

    data     = request.get_json(force=True)
    img_data = data.get("image", "")
    if "," in img_data:
        _, img_data = img_data.split(",", 1)

    try:
        img_bytes = base64.b64decode(img_data)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode frame")

        # Stateless call for finger-sign
        test_finger.SENTENCE  = []
        test_finger.LAST_TIME = 0.0

        _, sentence_list = test_finger.run_finger_on_frame(frame)
        char = sentence_list[-1] if sentence_list else ""
        return jsonify({ "action": char }), 200

    except Exception as e:
        print("❌ /process_finger_frame error:", e)
        return jsonify({ "action": "" }), 200

@app.route("/summary", methods=["POST"])
def summary():
    data      = request.get_json(force=True)
    sentences = data.get("sentences", [])
    try:
        text = generate_summary(sentences)
    except Exception as e:
        print("❌ /summary error:", e)
        text = ""
    return jsonify(summary=text), 200

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
