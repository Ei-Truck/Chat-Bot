from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
import os

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["chatbot_db"]
collection = db["documents"]

def embedding__files(folder_path=".\\app\\ai\\text"):

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            texts = f.read().strip()

        for text in texts.split("\n\n"):
            text = text.strip()
            embedding = model.encode(text)

            result = collection.update_one(
                {"filename": filename},  
                {
                    "$set": {
                        "text": text,
                        "embedding": embedding.tolist()
                    }
                },
                upsert=True  
            )
            if result.upserted_id:
                print(f"Inserido {text} com _id {result.upserted_id}")
            else:
                print(f"Atualizado documento do arquivo {text}\n")

def embedding__mongo():
    for doc in collection.find():
        if "text" not in doc:
            continue
        embedding = model.encode(doc["text"])
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": embedding.tolist()}},
            upsert=True
        )
        print(f"Atualizado {doc['_id']} com embedding.")
        
        
def search_embedding(question, top_k=1):
    question_embedded = model.encode(question).tolist()
    result = collection.find(
        {"embedding": {"$exists": True}}
        )
    results = []
    for doc in result:
        doc_embedding = doc["embedding"]
        distance = cosine_similarity([question_embedded], [doc_embedding])[0][0]
        results.append((distance, doc["text"]))
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return results    
