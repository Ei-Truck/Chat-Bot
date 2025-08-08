from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import numpy as np
from pymongo import MongoClient

# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Configurando a conexão com o Mongo
cliente = MongoClient("mongodb://localhost:27017/")

def hist(content:str,user_id:str):
    r.hset(f"histChat:{user_id}", mapping={
        "content": content,
        "genre": "hist",
        "embedding": model.encode(content).astype(np.float32).tobytes(),
    })

    q = Query(
        "*=>[KNN 3 @embedding $vec AS vector_distance]"
    ).return_field("score").dialect(2)

    query_text = 'Olá'
    res = r.ft("vector_idx").search(
        q, query_params={
            "vec": model.encode(query_text).astype(np.float32).tobytes()
        }
    )
    return res

print(hist("Olá, tudo bem?", "user123"))
