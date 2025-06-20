from flask import Blueprint, jsonify, request
from service import question_for_gemini

routes = Blueprint("routes", __name__)

@routes.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200

@routes.route("/chat", methods=["POST"])
def chat():
    data:dict = request.json
    question:str = data.get("message","")
    if not question:
        return jsonify({"error": "No message provided"}), 400

    answer:dict = question_for_gemini(question)
    if not answer:
        return jsonify({"error": "Failed to generate answer"}), 500

    return jsonify(answer), 200