from sentence_transformers import SentenceTransformer
import numpy as np

# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Histórico de chat
class ChatHistory:
    def __init__(self):
        self.history = []
        self.embeddings = []

    def armazenar_mensagem(self, user, msg):
        self.history.append({"user": user, "mensage": msg})
        embedding = model.encode(msg)
        self.embeddings.append(embedding)

    def search_history(self, query):
        query_embedding = model.encode(query)
        try:
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:3]
            return [self.history[i] for i in top_indices]
        except Exception as e:
            print("Erro ao buscar histórico:", e)
            return []
