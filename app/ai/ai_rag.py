import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import pymongo

# Carrega variáveis do .env
load_dotenv()

# Conexão MongoDB
mongo_host = os.getenv("CONNSTRING")
client_db = pymongo.MongoClient(mongo_host)
db = client_db["chatbot_db"]
collection = db["documents"]


# Função para retornar modelo HuggingFace
def get_model() -> HuggingFaceEmbeddings:
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model


# Função para gerar embeddings normalizados
def gerar_embeddings(text_list):
    if not text_list:
        return []

    model = get_model()
    embeddings = model.embed_documents(text_list)

    return embeddings


# Função para criar contexto de FAQ a partir de pergunta
def get_faq_context(question, txt_path="./app/ai/text/FAQ.txt") -> str:
    # Carrega o texto
    loader = TextLoader(txt_path, encoding="utf-8")
    docs = loader.load()

    # Divide em chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Usa o wrapper compatível
    embedding_function = get_model()

    # Cria o banco vetorial FAISS
    vector_db = FAISS.from_documents(chunks, embedding_function)

    # Busca os mais similares
    results = vector_db.similarity_search(question, k=6)

    # Junta o contexto
    context = "\n".join([doc.page_content for doc in results])
    return context


# Função para gerar embeddings e salvar no MongoDB
def embedding_files(file_path="./app/ai/text/FAQ.txt") -> None:
    # Evita duplicação se já existir
    if collection.count_documents({}) > 0:
        return

    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.read().strip()

    # Divide o texto em blocos separados por linha dupla
    chunks = [t.strip() for t in texts.split("\n\n") if t.strip()]

    # Gera embeddings em batch
    embeddings = gerar_embeddings(chunks)

    # Insere cada chunk como documento separado
    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        collection.update_one(
            {"filename": "FAQ.txt", "chunk_id": i},
            {"$set": {"text": text, "embedding": embedding}},
            upsert=True,
        )
