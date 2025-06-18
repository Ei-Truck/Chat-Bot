import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain_core.messages import HumanMessage 
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import re
import os


load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = chave_api)


# Inicia o modelo
model = genai.GenerativeModel("gemini-2.0-flash")

# Verificar pergunta
def verifica_pergunta(pergunta:str)-> str:
    llm = ChatGoogleGenerativeAI(google_api_key=chave_api,model="gemini-2.0-flash",temperature=0)
    prompt_avaliacao = "Você é um assistente que verifica se um texto contém linguagem ofensiva, discurso de ódio, calúnia ou difamação. Responda 'SIM' se contiver e 'NÃO' caso contrário. Seja estrito na sua avaliação."
    
    resposta_llm = llm.invoke([HumanMessage(content=prompt_avaliacao + "\n\nPergunta: " + pergunta)])
    return resposta_llm.content.strip()

# Rotina para enviar pergunta ao modelo
def responder_pergunta(pergunta: str) -> str:
    resposta = model.generate_content(pergunta)
    return resposta.text.strip()


# Utilizar o RAG
def rag_responder(pergunta: str) -> str:
    docs = []
    pasta = "text"
    for nome in os.listdir(pasta):
        if nome.endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            loader = TextLoader(caminho, encoding="utf-8")
            docs.extend(loader.load())
    rag = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    documentos = docs
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_divididos = splitter.split_documents(documentos)
    embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=chave_api,
    model="models/embedding-001"
    )

    db = FAISS.from_documents(docs_divididos, embeddings)
    # Cria o chain de pergunta-resposta com recuperação
    rag_chain = RetrievalQA.from_chain_type(
        llm = ChatGoogleGenerativeAI(google_api_key=chave_api,model="gemini-2.0-flash"),
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    resposta = rag_chain({"query": pergunta})
    return resposta['result'].strip()

# Verificar Resposta
def juiz_resposta(pergunta: str,resposta: str) -> str:
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    prompt_juiz = '''
    Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA para uma pergunta de aluno.

    Critérios:
    - A resposta está tecnicamente correta?
    - Está clara para o nível médio técnico?
    - O próximo passo sugerido está bem formulado?

    Se a resposta for boa, retorne a mesma resposta.
    Se tiver problemas, retorne uma versão melhorada.
    '''
    
    resposta_juiz = juiz([
    HumanMessage(content=prompt_juiz + "\n\nPergunta: " + pergunta + "\nResposta: " + resposta)
    ])    
    if "Aprovado" in resposta_juiz.content:
        return resposta.strip()
        
    else:
       return resposta_juiz.content.strip()
    import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS # Changed from langchain.vectorstores
from langchain_core.messages import HumanMessage # Changed from langchain.schema
from langchain_community.document_loaders import TextLoader # Changed from langchain.document_loaders
from langchain_text_splitters import CharacterTextSplitter # Changed from langchain.text_splitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import re
import os


# Carrega a variável de ambiente do arquivo .env (se você estiver usando .env)
load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = chave_api)


# Inicia o modelo
model = genai.GenerativeModel("gemini-2.0-flash")

# Verificar pergunta
def verifica_pergunta(pergunta:str)-> str:
    llm = ChatGoogleGenerativeAI(google_api_key=chave_api,model="gemini-2.0-flash",temperature=0)
    prompt_avaliacao = "Você é um assistente que verifica se um texto contém linguagem ofensiva, discurso de ódio, calúnia ou difamação. Responda 'SIM' se contiver e 'NÃO' caso contrário. Seja estrito na sua avaliação."
    
    resposta_llm = llm.invoke([HumanMessage(content=prompt_avaliacao + "\n\nPergunta: " + pergunta)])
    return resposta_llm.content.strip()

# Rotina para enviar pergunta ao modelo
def responder_pergunta(pergunta: str) -> str:
    resposta = model.generate_content(pergunta)
    return resposta.text.strip()


# Utilizar o RAG
def rag_responder(pergunta: str) -> str:
    docs = []
    pasta = "text"
    for nome in os.listdir(pasta):
        if nome.endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            loader = TextLoader(caminho, encoding="utf-8")
            docs.extend(loader.load())
    rag = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    documentos = docs
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_divididos = splitter.split_documents(documentos)
    embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=chave_api,
    model="models/embedding-001"
    )

    db = FAISS.from_documents(docs_divididos, embeddings)
    # Cria o chain de pergunta-resposta com recuperação
    rag_chain = RetrievalQA.from_chain_type(
        llm = ChatGoogleGenerativeAI(google_api_key=chave_api,model="gemini-2.0-flash"),
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    resposta = rag_chain({"query": pergunta})
    return resposta['result'].strip()

# Verificar Resposta
def juiz_resposta(pergunta: str,resposta: str) -> str:
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    prompt_juiz = '''
    Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA para uma pergunta de aluno.

    Critérios:
    - A resposta está tecnicamente correta?
    - Está clara para o nível médio técnico?
    - O próximo passo sugerido está bem formulado?

    Se a resposta for boa, retorne a mesma resposta.
    Se tiver problemas, retorne uma versão melhorada.
    '''
    
    resposta_juiz = juiz([
    HumanMessage(content=prompt_juiz + "\n\nPergunta: " + pergunta + "\nResposta: " + resposta)
    ])    
    if "Aprovado" in resposta_juiz.content:
        return resposta.strip()
        
    else:
       return resposta_juiz.content.strip()
    