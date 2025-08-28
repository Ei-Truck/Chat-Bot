from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np


# Resgatando o host do mongo
load_dotenv()

# Configura a conexão com o mongo
host_mongo = os.getenv("MONGO_HOST")
client = MongoClient(host=host_mongo)

db = client['hist_chat']

# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Histórico de chat
class ChatHistory:
    def __init__(self):
        pass
    
    def armazenar_mensagem(self,id_user,id_session,msg):
        try:
            collection = db[f'history_user_{id_user}']
            embedding = model.encode(msg)

            collection.update_one(
                {"_id": id_session}, 
                {
                    "$push": {
                        "embedding": embedding.tolist(),
                        "message": msg
                            }
                },
                upsert=True
            )
        except:
            pass
        
    def search_history(self, id_user,id_session, msg, top_k=3):
        try:
            collection = db[f"history_user_{id_user}"]
            query_embedding = np.array(model.encode(msg)).flatten()

            # pega embeddings e mensagens de TODAS as sessões do usuário
            cursor = collection.find({"_id":id_session}, {"_id": 1, "embedding": 1, "message": 1})
            all_embeddings, all_messages = [], []
            for doc in cursor:
                if "embedding" in doc and "message" in doc:
                    all_embeddings.extend(doc["embedding"])
                    all_messages.extend(doc["message"])
            if not all_embeddings:
                return "Ainda sem memória, chat novo"
            embeddings = np.array(all_embeddings)
            similarities = embeddings @ query_embedding
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [all_messages[i] for i in top_indices]
        except Exception as e:
            return "Ainda sem memória, chat novo"
