from flask import Flask
from .routes import routes

def api():
    app = Flask(__name__)
    app.register_blueprint(routes)
    return app