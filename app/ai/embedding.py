from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.json.path import Path
import numpy as np
import redis

# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Configurando a conexão com o Redis
r = redis.Redis(decode_responses=True, host='localhost', port=6379)

try:
    r.ft("vector_idx").dropindex(True)
except redis.exceptions.ResponseError:
    pass


def hist(content:str):
    r.hset("doc:1", mapping={
        "content": content,
        "genre": "hist",
        "embedding": model.encode(content).astype(np.float32).tobytes(),
    })

    q = Query(
        "*=>[KNN 3 @embedding $vec AS vector_distance]"
    ).return_field("score").dialect(2)

    res = r.ft("vector_idx").search(
        q, query_params={
            "vec": model.encode(query_text).astype(np.float32).tobytes()
        }
    )