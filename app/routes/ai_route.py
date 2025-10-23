from flask import Blueprint, jsonify, request
from prompt_toolkit.validation import ValidationError
from app.schemas.question_schema import AskSchema
from app.service.service_ai import question_for_gemini


# Cria um blueprint para organizar as rotas da aplicação
routes = Blueprint("routes", __name__)


# Rota de verificação de saúde da API
@routes.route("/health", methods=["GET"])
def health():
    """
    Endpoint de health check.
    Retorna status OK se a API estiver funcionando.
    """
    return jsonify({"status": "OK"}), 200


# Rota principal para enviar perguntas e receber respostas do AI
@routes.route("/chat", methods=["POST"])
def chat():
    """
    Recebe uma pergunta via POST, valida os dados e retorna a resposta do modelo AI.
    """
    # Pega os dados enviados no corpo da requisição
    data: dict = request.get_json()

    # Valida os dados recebidos usando o schema AskSchema
    try:
        validate_data = AskSchema().load(data)
    except ValidationError as err:
        # Retorna erro 400 caso os dados estejam no formato errado
        return jsonify({"error": str(err)}), 400

    # Chama a função que processa a pergunta e obtém a resposta do modelo
    answer: dict = question_for_gemini(
        validate_data["question"], validate_data["user_id"], validate_data["session_id"]
    )

    # Retorna erro 500 caso a função não consiga gerar uma resposta
    if not answer:
        return jsonify({"error": "Failed to generate answer"}), 500

    # Retorna a resposta gerada pelo modelo em formato JSON
    return jsonify(answer), 200
