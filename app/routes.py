from flask import Blueprint, jsonify

routes = Blueprint("routes", __name__)

@routes.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200