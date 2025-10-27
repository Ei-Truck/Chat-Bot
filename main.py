from app import api
import os
from flask_cors import CORS

app = api()
CORS(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
