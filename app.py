from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Import your OMR processing function
from omr import process_omr_bubbles

app = Flask(__name__, static_folder="static")
CORS(app)  # Allows your HTML frontend to talk to this server

# Make sure the static folder exists (for saving result images)
os.makedirs("static", exist_ok=True)


@app.route("/")
def index():
    """Serve the main HTML frontend."""
    return send_from_directory(".", "smartgrade.html")


@app.route("/grade", methods=["POST"])
def grade():
    """
    Receives the OMR image + answer key from the frontend.
    Returns the grading results as JSON.
    
    Expects a multipart/form-data POST with:
      - 'image'      : the uploaded OMR sheet image file
      - 'answer_key' : a JSON string like {"0": 1, "1": 3, "2": 0}
                       (question index → correct choice index, A=0 B=1 C=2 D=3)
      - 'choices'    : (optional) number of choices per question, default 4
    """

    # --- 1. Validate image was uploaded ---
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # --- 2. Read image as bytes ---
    image_bytes = image_file.read()

    # --- 3. Parse the answer key from the request ---
    import json
    try:
        raw_key = request.form.get("answer_key", "{}")
        # Keys come as strings from JSON, convert to integers
        # e.g. {"0": 1, "1": 3} → {0: 1, 1: 3}
        answer_key = {int(k): int(v) for k, v in json.loads(raw_key).items()}
    except Exception:
        return jsonify({"error": "Invalid answer key format."}), 400

    # --- 4. Parse choices per question (default 4: A B C D) ---
    try:
        choices = int(request.form.get("choices", 4))
    except ValueError:
        choices = 4

    # --- 5. Run your OMR processing function ---
    result = process_omr_bubbles(image_bytes, answer_key, choices_per_question=choices)

    # --- 6. Return the result ---
    if "error" in result:
        return jsonify(result), 422  # 422 = unprocessable image

    return jsonify(result), 200


if __name__ == "__main__":
    # debug=True auto-reloads when you save changes — turn off for final demo
    app.run(debug=True, port=5000)
