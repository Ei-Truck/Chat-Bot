from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Resgatando o host do mongo
load_dotenv()

# Configura a conexão com o mongo
conn_string = os.getenv("CONNSTRING")

client = MongoClient(conn_string)

db = client["hist_embedding"]
collection = db['history_embedded']


# Inicializando o modelo de embeddings
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")
    


# Embedding
def embedding_docs(docs):
    model = get_model()
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)
    for doc in docs:
        json_mongo = {
                "embedding":embeddings,
                "answer":texts
            }
        collection.insert_one(json_mongo)
    return True

def embedding_text(question,k=1):
    model = get_model()
    documentos = collection.find({})
    embedding_question = model.encode(question)
    for doc in documentos:
        texts = doc['embedding']
        similarities = cosine_similarity([embedding_question], texts).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]
        respostas = [doc['answer'][i] for i in top_indices]
    return respostas
    

def verifica_embedding(question):
    model = get_model()
    embedding_question = model.encode(question)
    embedding_question = np.array(embedding_question).reshape(1, -1).tolist()
    
    encontrado = collection.find(
            {"_id": 0,"embedding":1,"answer":1}
        ) 
    
    maior_similaridade = 0
    
    for x in encontrado:
        similarities = cosine_similarity(x[0], embedding_question).flatten()
        if similarities > maior_similaridade:
            maior_similaridade = similarities
            user_answer = x[0]
            
    if maior_similaridade!=0:
        encontrado = collection.find(
            {"question":user_answer},  {"_id": 0,"id_question": 0,"question":0}
        ) 
    
        for doc in encontrado:
            return doc["answer"]
    return None