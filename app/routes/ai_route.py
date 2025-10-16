from flask import Blueprint, jsonify, request
from prompt_toolkit.validation import ValidationError
from app.schemas.question_schema import AskSchema
from app.service.service_ai import question_for_gemini


routes = Blueprint("routes", __name__)


@routes.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200


@routes.route("/chat", methods=["POST"])
def chat():
    data: dict = request.get_json()
    try:
        validate_data = AskSchema().load(data)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

    answer: dict = question_for_gemini(
        validate_data["question"], validate_data["user_id"], validate_data["session_id"]
    )

    if not answer:
        return jsonify({"error": "Failed to generate answer"}), 500

    return jsonify(answer), 200
