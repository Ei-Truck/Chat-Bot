import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from app.ai.embedding import embedding_text
import os

load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=chave_api)

# Inicia o modelo e memoria
model = genai.GenerativeModel("gemini-2.0-flash")


# Verificar pergunta
def verifica_pergunta(pergunta: str) -> str:
    llm = ChatGoogleGenerativeAI(
        google_api_key=chave_api,
        model="gemini-2.0-flash",
        temperature=0
    )
    prompt_avaliacao = (
        "Você é um assistente que verifica se um texto contém "
        "linguagem ofensiva, discurso de ódio, calúnia ou difamação. "
        "Responda 'SIM' se contiver e 'NÃO' caso contrário. Seja estrito na sua avaliação."
    )

    resposta_llm = llm.invoke(
        [HumanMessage(content=prompt_avaliacao + "\n\nPergunta: " + pergunta)]
    )
    return resposta_llm.content.strip()


# Responder com o gemini
def gemini_resp(pergunta: str) -> str:
    normal_chat = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        google_api_key=chave_api
    )
    prompt_gemini = (
        f"Você é um agente de perguntas e respostas da empresa EiTruck.\n"
        f"Um usuário do chat fez a seguinte pergunta: {pergunta}\n"
        "Por favor, responda de forma precisa, detalhada e elaborada, focando em fornecer a melhor resposta possível.\n"
        "Resuma a resposta ao máximo, focando em ser objetivo.\n"
        "Mantenha o foco na pergunta e não desvie do assunto."
    )
    response = normal_chat([HumanMessage(content=prompt_gemini)])
    return response.content.strip()


# Utilizar o RAG
def rag_responder(pergunta: str) -> str:
    docs = []
    pasta = "app/ai/text/"
    for nome in os.listdir(pasta):
        if nome.endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            loader = TextLoader(caminho, encoding="utf-8")
            docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    docs_divididos = splitter.split_documents(docs)
    return embedding_text(docs_divididos, pergunta, 1)


# Verificar Resposta
def juiz_resposta(pergunta: str, resposta: str) -> str:
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        google_api_key=chave_api
    )

    prompt_juiz = (
        "Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA.\n\n"
        "Critérios:\n"
        "- A resposta está tecnicamente correta?\n"
        "- Está clara para o nível médio técnico?\n"
        "- O próximo passo sugerido está bem formulado?\n\n"
        "Se não souber como responder:\n"
        "- procure a melhor resposta possível para a pergunta proposta.\n"
        "- reavalie a resposta.\n"
        "- não retorne a sua avaliação anterior à correção.\n\n"
        "Se a resposta for boa:\n"
        "- defina o status como 'Aprovado'\n"
        "- retorne o porquê que a resposta é boa.\n\n"
        "Se tiver problemas:\n"
        "- defina o status como Reprovado\n"
        "- proponha uma versão melhorada.\n\n"
        "Independente da ocasião, retorne um JSON com os campos:\n"
        "- status: 'Aprovado' ou 'Reprovado'\n"
        "- question: a pergunta realizada\n"
        "- answer: a resposta original antes da correção\n"
        "- judgmentAnswer: a resposta do juiz\n\n"
        "A resposta deve estar no formato correto do JSON."
    )

    resposta_juiz = juiz(
        [HumanMessage(
            content=prompt_juiz + "\n\nPergunta:" + pergunta + "\nResposta:" + resposta
        )]
    )

    resposta_juiz = resposta_juiz.content.strip()
    if resposta_juiz.startswith("```json"):
        resposta_juiz = resposta_juiz[len("```json"):].rstrip("```").strip()

    return resposta_juiz
