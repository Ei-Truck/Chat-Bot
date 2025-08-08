from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 


# Resgatando o host do mongo
load_dotenv()

# Configura a conexão com o mongo
host_mongo = os.getenv("MONGO_HOST")
client = MongoClient(host=host_mongo)

# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Embedding
def embedding_text(content:str,text,user_id:str):
    embedding = model.encode(content)
    similarity = cosine_similarity([embedding])[0][0]
    return similarity
    

docs = []
pasta = "app/ai/text/"
for nome in os.listdir(pasta):
    if nome.endswith(".txt"):
        caminho = os.path.join(pasta, nome)
        loader = TextLoader(caminho, encoding="utf-8")
        docs.extend(loader.load())
        documentos = docs
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_divididos = splitter.split_documents(documentos)
        print(embedding_text(docs_divididos,"O que é telemetria?",1))