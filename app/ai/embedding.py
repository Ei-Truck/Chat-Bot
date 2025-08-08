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
def embedding_text(docs, question, k):
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)
    embedding_question = model.encode(question)
    embeddings = np.array(embeddings)                  
    embedding_question = np.array(embedding_question).reshape(1, -1)  
    similarities = cosine_similarity(embeddings, embedding_question).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(texts[0], similarities[0]) for i in top_k_indices]

def armazenar_vetores(id,question):
    print()  

docs = []
pasta = "app/ai/text/"
for nome in os.listdir(pasta):
    if nome.endswith(".txt"):
        caminho = os.path.join(pasta, nome)
        loader = TextLoader(caminho, encoding="utf-8")
        docs.extend(loader.load())
        documentos = docs
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs_divididos = splitter.split_documents(documentos)
        texto = embedding_text(docs_divididos,"O que é telemetria?",1)[0][0]
        print(texto)