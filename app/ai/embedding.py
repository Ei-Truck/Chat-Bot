from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Resgatando o host do mongo
load_dotenv()

# Configura a conexão com o mongo
host_mongo = os.getenv("MONGO_HOST")
client = MongoClient(host=host_mongo)
db = client['hist_embedding']
collection = db['hist_embedded']


# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embedding
def embedding_text(docs, question, k):
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)
    embedding_question = model.encode(question)
    embeddings = np.array(embeddings)                  
    embedding_question = np.array(embedding_question).reshape(1, -1)  
    similarities = cosine_similarity(embeddings, embedding_question).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(texts[0], similarities[0]) for i in top_k_indices]


def historico_gemini(question,answer):
    embedding_question = model.encode(question)
    embedding_question = np.array(embedding_question).reshape(1, -1)
    json_mongo = {
                "question":embedding_question,
                "answer":answer
            }
    collection.insert_one(json_mongo)
    return True

def verifica_embedding(question):
    embedding_question = model.encode(question)
    embedding_question = np.array(embedding_question).reshape(1, -1)
    
    encontrado = collection.find(
            {"_id": 0,"id_question": 0,"question":0}
        ) 
    
    maior_similaridade = 0
    
    for x in encontrado:
        similarities = cosine_similarity(x[0], embedding_question).flatten()
        if similarities > maior_similaridade:
            maior_similaridade = similarities
            user_question = x[0]
    
    encontrado = collection.find(
        {"question":user_question},  {"_id": 0,"id_question": 0,"question":0}
        ) 
    
    for doc in encontrado:
        return doc["answer"]
    return None