import os
import base64
import cv2
import numpy as np
from flask           import Flask, request, jsonify, send_from_directory
from flask_cors      import CORS
from test_5          import run_test_on_frame, actions as MODEL_ACTIONS
from test_gpt        import generate_summary
import test_finger

app = Flask(
    __name__,
    static_folder="frontend/build",
    static_url_path=""    # serve React at /
)
CORS(app)

@app.route("/actions", methods=["GET"])
def get_actions():
    return jsonify(actions=MODEL_ACTIONS.tolist())

@app.route("/process_frame", methods=["OPTIONS", "POST"])
def process_frame():
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
            raise ValueError("Could not decode frame")

        _, probs, sentence_list = run_test_on_frame(frame)
        return jsonify({
            "probabilities": probs.tolist(),
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
            raise ValueError("Could not decode frame")

        # Reset server-side finger state so each request is fresh
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
    data = request.get_json(force=True)
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
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
