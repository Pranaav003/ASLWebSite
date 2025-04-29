# backend/app.py

import os
import base64
import cv2
import numpy as np
from flask             import Flask, request, jsonify, send_from_directory
from flask_cors        import CORS
from test_5            import run_test_on_frame, actions as MODEL_ACTIONS
from test_gpt          import generate_summary
from test_finger       import run_finger_on_frame

app = Flask(
    __name__,
    static_folder="frontend/build",
    static_url_path=""    # serve React build at /
)
CORS(app)

# In‐memory stores
LAST_SENTENCES = []
LAST_FINGER_SENTENCE = []

@app.route("/actions", methods=["GET"])
def get_actions():
    """
    Return the list of gesture labels from your main ASL model.
    """
    return jsonify(actions=MODEL_ACTIONS.tolist())

@app.route("/process_frame", methods=["OPTIONS", "POST"])
def process_frame():
    """
    Accepts a base64 JPEG from the browser, decodes and runs run_test_on_frame,
    and returns probabilities + the concatenated sentence.
    """
    global LAST_SENTENCES

    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(force=True)
    img_data = data.get("image", "")
    if "," in img_data:
        _, img_data = img_data.split(",", 1)

    try:
        img_bytes = base64.b64decode(img_data)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image frame")

        try:
            _, probs, sentence_list = run_test_on_frame(frame)
            LAST_SENTENCES = sentence_list
        except Exception as e:
            # log the error but continue
            print("⚠️ Gesture model error:", e)
            probs = []
            sentence_list = LAST_SENTENCES

        return jsonify({
            "probabilities": probs.tolist() if hasattr(probs, "tolist") else probs,
            "action": " ".join(sentence_list)
        })

    except Exception as e:
        print("❌ /process_frame error:", e)
        return jsonify({
            "probabilities": [],
            "action": ""
        }), 200

@app.route("/process_finger_frame", methods=["OPTIONS", "POST"])
def process_finger_frame():
    """
    Accepts a base64 JPEG from the browser, decodes and runs run_finger_on_frame,
    and returns the accumulated finger‐sign sentence.
    """
    global LAST_FINGER_SENTENCE

    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(force=True)
    img_data = data.get("image", "")
    if "," in img_data:
        _, img_data = img_data.split(",", 1)

    try:
        img_bytes = base64.b64decode(img_data)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image frame")

        try:
            annotated_frame, sentence = run_finger_on_frame(frame)
            LAST_FINGER_SENTENCE = sentence
        except Exception as e:
            print("⚠️ Finger model error:", e)
            sentence = LAST_FINGER_SENTENCE

        return jsonify({
            "action": "".join(sentence)
        })

    except Exception as e:
        print("❌ /process_finger_frame error:", e)
        return jsonify(action=""), 200

@app.route("/summary", methods=["GET"])
def summary():
    """
    Uses the last detected ASL gesture sentence to generate example sentences
    via ChatGPT.
    """
    try:
        summary_text = generate_summary(LAST_SENTENCES)
    except Exception as e:
        print("❌ /summary error:", e)
        summary_text = ""
    return jsonify(summary=summary_text)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    """
    Serve React static files. All unknown routes return index.html.
    """
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    # debug=True for local development; disable in production
    app.run(host="0.0.0.0", port=port, debug=True)
