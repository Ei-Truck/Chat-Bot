from langchain_mongodb import MongoDBChatMessageHistory
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np


# Resgatando o host do mongo
load_dotenv()

# Configura a conexão com o mongo
conn_string = os.getenv("CONNSTRING")

client = MongoClient(conn_string)


db = client['hist_chat']

# Inicializando o modelo de embeddings e histórigo
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Instanciando histórico
def get_chat_hist(id_user,id_session):
    chat_hist = MongoDBChatMessageHistory(
        connection_string=conn_string,
        session_id=str(id_session),
        database_name='hist_chat',
        collection_name=f'history_user_{id_user}',
        create_index=True
        
    )
    return chat_hist


# Metodos

def get_session_history(session_id):
    history = get_chat_hist(user_id=None, session_id=session_id)
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            messages.append({"role": "user", "content": msg.get("mensagem", "")})
        elif isinstance(msg, str):
            messages.append({"role": "user", "content": msg})
    return messages

def salvar_historico(id_user,session_id,question,answer):
    
    collection = db[f'history_user_{id_user}']
    collection.insert_one({
        "session_id": session_id,
        "interaction": {
            "input": question,
            "response": answer
        }
    })
    print()

# Histórico de chat
class ChatHistory:
    def __init__(self):
        pass
    
    def armazenar_mensagem(self,id_user,id_session,msg):
        try:
            collection = db[f'history_user_{id_user}']
            model = get_model()
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
            model = get_model()
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
