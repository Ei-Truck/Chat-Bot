from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import os

load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
mongo_host = os.getenv("CONNSTRING")


# Verificar pergunta
def verifica_pergunta(pergunta: str) -> str:
    llm = ChatGoogleGenerativeAI(google_api_key=chave_api, model="gemini-1.5-flash", temperature=0)
    prompt_avaliacao = (
        "Você é um assistente que verifica se um texto contém "
        "linguagem ofensiva, discurso de ódio, calúnia ou difamação. "
        "Responda 'SIM' se contiver e 'NÃO' caso contrário. Seja estrito na sua avaliação."
    )

    resposta_llm = llm.invoke(
        [HumanMessage(content=prompt_avaliacao + "\n\nPergunta: " + pergunta)]
    )
    return resposta_llm.content.strip()


def get_session_history(user_id, session_id) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=f"{user_id}_{session_id}",
        connection_string=mongo_host,
        database_name="chatbot_db",
        collection_name="chat_histories",
    )


def gemini_resp(user_id, session_id, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.7, top_p=0.95, google_api_key=chave_api
    )
    prompt_gemini = os.path.join(os.path.dirname(__file__), "prompt_gemini.txt")
    system_prompt = (
        "system",
        prompt_gemini
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{human}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ]
    )
    shots = []
    fewshots = FewShotChatMessagePromptTemplate(examples=shots, example_prompt=example_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            fewshots,
            MessagesPlaceholder("chat_history"),
            ("human", "{usuario}"),
        ]
    )
    base_chain = prompt | llm | StrOutputParser()
    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="usuario",
        history_messages_key="chat_history",
    )
    if question.lower() in ("sair", "end", "fim", "tchau", "bye"):
        return "Encerrando o chat."
    try:
        resposta = chain.invoke(
            {"usuario": question}, config={"configurable": {"session_id": session_id}}
        )
        return resposta
    except Exception as e:
        return f"Não foi possível responder: {e}"


# Verificar Resposta
def juiz_resposta(pergunta: str, resposta: str) -> str:
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.5, google_api_key=chave_api
    )

    prompt_juiz = os.path.join(os.path.dirname(__file__), "prompt_juiz.txt")

    resposta_juiz = juiz(
        [HumanMessage(content=prompt_juiz + "\n\nPergunta:" + pergunta + "\nResposta:" + resposta)]
    )

    resposta_juiz = resposta_juiz.content.strip()
    if resposta_juiz.startswith("```json"):
        resposta_juiz = resposta_juiz[len("```json"):].rstrip("```").strip()

    return resposta_juiz
