from flask import Blueprint, jsonify, request

routes = Blueprint("routes", __name__)

@routes.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200

@routes.route("/version", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message","")
    if not question:
        return jsonify({"error": "No message provided"}), 400

    answer = question_for_gemini(question)
    if not answer:
        return jsonify({"error": "Failed to generate answer"}), 500

    return jsonify({"answer": answer}), 200