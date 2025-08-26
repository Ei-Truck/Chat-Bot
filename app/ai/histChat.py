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
    
#Historico de chat no Redis  
class ChatHistory:
    def __init__(self):
        self.history = []
        self.embeddings = []
        
    def armazenar_mensagem(self,id_user,id_session,msg):
        self.history.append({"user":f"{id_user}_{id_session}","mensage":msg})
        collection = db[f'history_user_{id_user}_{id_session}']
        json = {
            "_id":f"{id_user}_{id_session}",
            "mensage":msg
        }
        collection._insert_one(doc=json,session=id_session)
        embedding = model.encode(msg)
        self.embeddings.append(embedding)
        
    def search_history(id_user,id_session,self,query):
        query_embedding = model.encode(query)
        try:
            similarities = np.dot(self.embeddings,query_embedding)
            top_indicies = np.argsort(similarities)[::-1][:3]
            return [self.history[i] for i in top_indicies]
        except:
            return 0