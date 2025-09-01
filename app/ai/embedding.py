from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
conn_string = os.getenv("CONNSTRING")
client = MongoClient(conn_string)
db = client["hist_embedding"]
collection = db["history_embedded"]

def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

def embedding_docs(docs):
    try:
        model = get_model()
        texts = [doc.page_content for doc in docs]
        embeddings = model.encode(texts)
        inserted_count = 0
        for text, emb in zip(texts, embeddings):
            doc_data = {
                "embedding": emb.tolist() if hasattr(emb, "tolist") else emb,
                "answer": text
            }
            collection.insert_one(doc_data)
            inserted_count += 1
        return inserted_count
    except Exception as e:
        print(f"Erro ao inserir embeddings: {e}")
        return 0

def embedding_text(question, k=1):
    model = get_model()
    embedding_question = model.encode(question).reshape(1, -1)
    documentos = list(collection.find({}))
    if not documentos:
        return []
    similarities_global = []
    for doc in documentos:
        doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
        sim = cosine_similarity(embedding_question, doc_embedding)[0][0]
        similarities_global.append((sim, doc["answer"]))
    similarities_global.sort(key=lambda x: x[0], reverse=True)
    top_respostas = [answer for sim, answer in similarities_global[:k]]
    return top_respostas



def verifica_embedding(question):
    model = get_model()
    embedding_question = model.encode(question).reshape(1, -1)
    documentos = list(collection.find({}))
    if not documentos:
        return None
    maior_similaridade = 0
    resposta_mais_proxima = None
    for doc in documentos:
        doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
        sim = cosine_similarity(embedding_question, doc_embedding)[0][0]
        if sim > maior_similaridade:
            maior_similaridade = sim
            resposta_mais_proxima = doc["answer"]
    return resposta_mais_proxima if maior_similaridade > 0 else None
