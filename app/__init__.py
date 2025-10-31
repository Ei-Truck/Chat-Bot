from flask import Flask
from app.routes.ai_route import routes


# Função que cria e configura a aplicação Flask
def api():
    # Cria a instância da aplicação Flask
    app = Flask(__name__)

    # Registra o blueprint contendo as rotas da API (ex: /chat, /health)
    app.register_blueprint(routes)

    # Retorna a aplicação pronta para ser executada
    return app
