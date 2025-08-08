from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Resgatando o host do mongo
load_dotenv()

# Configura a API key
host_mongo = os.getenv("MONGO_HOST")


# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Configurando a conexão com o Mongo
client = MongoClient(host=host_mongo)

# Embedding
def embedding_text(content:str,user_id:str):
    

    q = Query(
        "*=>[KNN 3 @embedding $vec AS vector_distance]"
    ).return_field("score").dialect(2)

    
    