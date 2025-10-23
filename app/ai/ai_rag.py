import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import pymongo

# Carrega variáveis do arquivo .env, permitindo acessar credenciais e configurações sensíveis
load_dotenv()

# Conexão com o banco de dados MongoDB usando a string de conexão definida no .env
mongo_host = os.getenv("CONNSTRING")
client_db = pymongo.MongoClient(mongo_host)
db = client_db["chatbot_db"]  # Seleciona o banco de dados
collection = db["documents"]  # Define a coleção onde os dados serão armazenados


# Função responsável por retornar o modelo de embeddings da HuggingFace
def get_model() -> HuggingFaceEmbeddings:
    # O modelo 'all-MiniLM-L6-v2' é eficiente para gerar vetores semânticos
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model


# Função que gera embeddings normalizados a partir de uma lista de textos
def gerar_embeddings(text_list):
    # Caso a lista esteja vazia, retorna uma lista vazia
    if not text_list:
        return []

    # Carrega o modelo de embeddings
    model = get_model()

    # Gera os embeddings para todos os textos na lista
    embeddings = model.embed_documents(text_list)

    return embeddings


# Função que cria um contexto de FAQ com base em uma pergunta específica
def get_faq_context(question, txt_path="./app/ai/text/FAQ.txt") -> str:
    # Carrega o arquivo de texto contendo as FAQs
    loader = TextLoader(txt_path, encoding="utf-8")
    docs = loader.load()

    # Divide o texto em blocos menores (chunks) para processamento mais eficiente
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Define a função de embeddings (modelo HuggingFace)
    embedding_function = get_model()

    # Cria o banco vetorial FAISS com base nos chunks e no modelo de embeddings
    vector_db = FAISS.from_documents(chunks, embedding_function)

    # Realiza a busca pelos chunks mais similares à pergunta feita
    results = vector_db.similarity_search(question, k=6)

    # Junta o conteúdo dos chunks mais relevantes em um único contexto textual
    context = "\n".join([doc.page_content for doc in results])
    return context


# Função que gera embeddings dos arquivos e os salva no MongoDB
def embedding_files(file_path="./app/ai/text/FAQ.txt") -> None:
    # Evita duplicar dados caso a coleção já tenha documentos
    if collection.count_documents({}) > 0:
        return

    # Abre o arquivo de texto e lê seu conteúdo completo
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.read().strip()

    # Divide o conteúdo em blocos separados por linha dupla
    chunks = [t.strip() for t in texts.split("\n\n") if t.strip()]

    # Gera embeddings em lote para todos os blocos de texto
    embeddings = gerar_embeddings(chunks)

    # Insere cada bloco (chunk) como um documento individual na coleção do MongoDB
    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        collection.update_one(
            {"filename": "FAQ.txt", "chunk_id": i},  # Filtro: identifica o chunk
            {
                "$set": {"text": text, "embedding": embedding}
            },  # Atualiza ou insere o dado
            upsert=True,  # Cria o documento caso ele não exista
        )
