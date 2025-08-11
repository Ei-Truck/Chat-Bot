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


def historico_gemini(id,question,answer):
    collection = db[f'hist_usr_{id}']
    embeddings = model.encode(answer)
    embedding_question = model.encode(question)
    embeddings = np.array(embeddings).reshape(1, -1)
    embedding_question = np.array(embedding_question).reshape(1, -1)
    similarities = cosine_similarity(embeddings, embedding_question).flatten()
    encontrado = collection.find_one({"question":float(similarities[0])},{ "_id": 0 })
    if encontrado == None:
        json_mongo = {
                    "question":float(similarities[0]),
                    "answer":answer
                }
        collection.insert_one(json_mongo)
        encontrado = collection.find_one({"question":float(similarities[0])},{ "_id": 0 })
    return encontrado