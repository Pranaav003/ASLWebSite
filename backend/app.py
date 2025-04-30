# backend/app.py

import os
import base64
import cv2
import numpy as np
from flask       import Flask, request, jsonify, send_from_directory
from flask_cors  import CORS
from collections import deque

from init_1       import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
from test_5       import model, actions as MODEL_ACTIONS
from test_gpt     import generate_summary
import test_finger

app = Flask(
    __name__,
    static_folder="frontend/build",
    static_url_path=""
)
CORS(app)

# Constants for ASL gesture buffering & smoothing
SEQ_LENGTH = 30
SMOOTH_WIN = 10

# In-memory per-client state
ASL_CLIENTS = {}  # client_id -> { sequence, predictions, sentence, holistic, last_probs }

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
    """Return the list of gesture action labels."""
    return jsonify(actions=MODEL_ACTIONS.tolist())

@app.route("/process_frame", methods=["OPTIONS", "POST"])
def process_frame():
    """
    Decode a base64 image, run Mediapipe+LSTM inference for the given client,
    and return probabilities + accumulated sentence.
    Guards against empty/malformed POSTs by returning last known state.
    """
    if request.method == "OPTIONS":
        return "", 200

    data      = request.get_json(force=True)
    client_id = data.get("clientId")
    img_data  = data.get("image", "")

    if not client_id:
        return jsonify(error="Missing clientId"), 400

    # Retrieve or create this client's state
    state = get_client_state(client_id)

    # If no image payload, return last known output
    if not img_data or not isinstance(img_data, str):
        return jsonify({
            "probabilities": state.get("last_probs", []),
            "action":        " ".join(state["sentence"])
        }), 200

    # Strip data URL prefix if present
    if "," in img_data:
        _, img_data = img_data.split(",", 1)

    try:
        # Decode image
        img_bytes = base64.b64decode(img_data)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode frame")

        # Mediapipe detection & drawing
        image, results = mediapipe_detection(frame, state["holistic"])
        if results:
            draw_styled_landmarks(image, results)

        # Extract keypoints and update sequence buffer
        keypoints = extract_keypoints(results) if results else np.zeros(1662)
        state["sequence"].append(keypoints)

        # Default empty probabilities
        probs = np.zeros(len(MODEL_ACTIONS))
        # Once we have a full sequence, run model predict
        if len(state["sequence"]) == SEQ_LENGTH:
            preds = model.predict(
                np.expand_dims(state["sequence"], axis=0)
            )[0]
            idx = int(np.argmax(preds))
            state["predictions"].append(idx)

            # Smoothing & threshold logic
            if (
                len(state["predictions"]) >= SMOOTH_WIN
                and state["predictions"][-SMOOTH_WIN:].count(idx) == SMOOTH_WIN
                and preds[idx] > 0.8
            ):
                word = MODEL_ACTIONS[idx]
                if not state["sentence"] or state["sentence"][-1] != word:
                    state["sentence"].append(word)
            probs = preds

        # Save last_probs for empty-POST fallback
        state["last_probs"] = probs.tolist()

        return jsonify({
            "probabilities": probs.tolist(),
            "action":        " ".join(state["sentence"])
        })

    except Exception as e:
        # On error, return last known state instead of crashing
        print("❌ /process_frame error:", e)
        return jsonify({
            "probabilities": state.get("last_probs", []),
            "action":        " ".join(state["sentence"])
        }), 200

@app.route("/process_finger_frame", methods=["OPTIONS", "POST"])
def process_finger_frame():
    """
    Decode base64 image, run finger-sign LSTM inference statelessly, and
    return the latest character. Resets internal buffer each call.
    """
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

        # Make each call stateless for fingers
        test_finger.SENTENCE  = []
        test_finger.LAST_TIME = 0.0

        _, sentence_list = test_finger.run_finger_on_frame(frame)
        char = sentence_list[-1] if sentence_list else ""
        return jsonify({ "action": char })

    except Exception as e:
        print("❌ /process_finger_frame error:", e)
        return jsonify({ "action": "" }), 200

@app.route("/summary", methods=["POST"])
def summary():
    """
    Accepts a JSON body {"sentences": [...]} and returns a ChatGPT-generated
    summary/examples string.
    """
    data      = request.get_json(force=True)
    sentences = data.get("sentences", [])
    try:
        text = generate_summary(sentences)
    except Exception as e:
        print("❌ /summary error:", e)
        text = ""
    return jsonify(summary=text)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    """
    Serve React build artifacts for all routes not matched above.
    """
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    # debug=True is fine locally but disable in production
    app.run(host="0.0.0.0", port=port, debug=True)
