from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pymongo
import os

load_dotenv()

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

mongo_host = os.getenv("CONNSTRING")


client = pymongo.MongoClient(mongo_host)
db = client["chatbot_db"]
collection = db["documents"]


def embedding_files(folder_path=".\\app\\ai\\text"):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            texts = f.read().strip()
        for text in texts.split("\n\n"):
            text = text.strip()
            embedding = model.encode(text)
            collection.update_one(
                {"filename": filename},
                {"$set": {"text": text, "embedding": embedding.tolist()}},
                upsert=True,
            )


def search_embedding(question, top_k=1):
    question_embedded = model.encode(question).tolist()
    result = collection.find({"embedding": {"$exists": True}})
    results = []
    for doc in result:
        doc_embedding = doc["embedding"]
        distance = cosine_similarity(
            [question_embedded], [doc_embedding])[0][0]
        results.append((distance, doc["text"]))
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return results
