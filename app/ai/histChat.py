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
            collection = db[f'history_user_{id_user}_{id_session}']
            embedding = model.encode(msg)

            collection.update_one(
                {"_id": f"{id_user}_{id_session}"}, 
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
        
    def search_history(self,id_user,id_session,msg):
        collection = db[f'history_user_{id_user}_{id_session}']
        query_embedding = model.encode(msg)
        try:
            doc = collection.find_one({"_id": f"{id_user}_{id_session}"}, {"_id": 0, "embedding": 1,"message": 1})
            if not doc:
                return "Ainda sem memória, chat novo" 
            embeddings = np.array(doc["embedding"])
            similarities = np.dot(embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:3]
            return [doc["message"][i] for i in top_indices]
        except:
            return "Ainda sem memória, chat novo"
