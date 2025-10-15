import os
import numpy as np
from google.generativeai import types
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pymongo

# Carrega variáveis do .env
load_dotenv()

# Conexão MongoDB
mongo_host = os.getenv("CONNSTRING")
client_db = pymongo.MongoClient(mongo_host)
db = client_db["chatbot_db"]
collection = db["documents"]

api_gemini = genai.Client(api_key=os.getenv("EMBEDDING_API_KEY"))


def gerar_embeddings(text_list):
    
    if not text_list:
        return []

    result = api_gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=text_list,
        config=types.EmbedContentConfig(output_dimensionality=512),
    )

    normed_embeddings = []
    for embedding_obj in result.embeddings:
        embedding_values_np = np.array(embedding_obj.values)
        normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
        normed_embeddings.append(normed_embedding.tolist())

    return normed_embeddings


def embedding_files(folder_path="./app/ai/text") -> None:

    if collection.count_documents({}) > 0:
        return

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            texts = f.read().strip()

        for text in texts.split("\n\n"):
            text = text.strip()
            embedding = gerar_embeddings([text])[0]
            collection.update_one(
                {"filename": filename},
                {"$set": {"text": text, "embedding": embedding}},
                upsert=True,
            )


def search_embedding(question, top_k=1) -> list:
    
    print("\n\n\nProcurando embedding da pergunta...\n\n\n")

    
    question_embedding = gerar_embeddings([question])[0]
    result = collection.find({"embedding": {"$exists": True}})
    results = []

    for doc in result:
        doc_embedding = np.array(doc["embedding"])
        distance = cosine_similarity([question_embedding], [doc_embedding])[0][0]
        results.append((distance, doc["text"]))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return results
