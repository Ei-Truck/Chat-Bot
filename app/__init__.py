from flask import Flask
from app.routes.ai_route import routes


def api():
    app = Flask(__name__)
    app.register_blueprint(routes)
    return app
