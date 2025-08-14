import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain_core.messages import HumanMessage 
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from dotenv import load_dotenv
from app.ai.embedding import embedding_text
import os


load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = chave_api)


# Inicia o modelo
model = genai.GenerativeModel("gemini-2.0-flash")

# Verificar pergunta
def verifica_pergunta(pergunta:str)-> str:
    llm = ChatGoogleGenerativeAI(
        google_api_key=chave_api,
        model="gemini-2.0-flash",
        temperature=0
    )
    prompt_avaliacao = "Você é um assistente que verifica se um texto contém linguagem ofensiva, discurso de ódio, calúnia ou difamação. Responda 'SIM' se contiver e 'NÃO' caso contrário. Seja estrito na sua avaliação."

    resposta_llm = llm.invoke([HumanMessage(content=prompt_avaliacao + "\n\nPergunta: " + pergunta)])
    return resposta_llm.content.strip()

# Responder com o gemini
def gemini_resp(pergunta: str) -> str:
    normal_chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    prompt_gemini = f"""
        Você é um agente de perguntas e respostas da empresa EiTruck.
        Um usuário do chat fez a seguinte pergunta: {pergunta}
        Por favor, responda de forma precisa, detalhada e elaborada, focando em fornecer a melhor resposta possível.
        Resuma a resposta ao máximo, focando em ser objetivo.
        Mantenha o foco na pergunta e não desvie do assunto.
        """
    response = normal_chat([HumanMessage(content=prompt_gemini)])
    return response.content.strip()

# Utilizar o RAG
def rag_responder(id,pergunta: str) -> str:
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
    return embedding_text(docs_divididos,pergunta,1) 

# Verificar Resposta
def juiz_resposta(pergunta: str,resposta: str) -> str:
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    prompt_juiz = f'''
    Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA.

    Critérios:
        - A resposta está tecnicamente correta?
        - Está clara para o nível médio técnico?
        - O próximo passo sugerido está bem formulado?

    Se não souber como responder:
        - procure a melhor resposta possível para a pergunta proposta.
        - reavalie a resposta.
        - não retorne a sua avaliação anterior à correção.

    Se a resposta for boa:
        - defina o status como "Aprovado"
        - retorne o porquê que a resposta é boa.

    Se tiver problemas:
        - defina o status como Reprovado
        - proponha uma versão melhorada.

    Independente da ocasião, retorne um JSON com os campos:
        - status: "Aprovado" ou "Reprovado"
        - question: a pergunta realizada
        - answer: a resposta original antes da correção
        - judgmentAnswer: a resposta do juiz

    A resposta deve estar no formato correto do JSON.
    Remova o json do inicio da resposta   
'''
    
    
    resposta_juiz = juiz([
    HumanMessage(
        content=prompt_juiz + "\n\nPergunta: " + pergunta + "\nResposta: " + resposta
    )
    ])
    resposta_juiz = resposta_juiz.content.strip()
    if resposta_juiz.startswith("```json"):
        resposta_juiz = resposta_juiz[len("```json"):].rstrip("```").strip()
    return resposta_juiz
    
    