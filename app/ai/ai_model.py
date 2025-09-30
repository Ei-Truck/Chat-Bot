from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from zoneinfo import ZoneInfo
from datetime import datetime
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import os

load_dotenv()

TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()
hour = datetime.now(TZ).time()


# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
mongo_host = os.getenv("CONNSTRING")

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=chave_api
)

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


# Roteador para localizar agente especialista
def roteador_eitruck(user_id, session_id) -> RunnableWithMessageHistory: 
    session=f"{user_id}_{session_id}"
    with open("./app/ai/text/prompt_juiz.txt", "r", encoding="utf-8") as f:
        prompt_roteador_text = f.read()  

    system_prompt = (
        "system",
        prompt_roteador_text
    )
    prompt_roteador = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ]
    )
    shots_roteador = [
        {
            "human": "Oi, tudo bem?",
            "ai": "Olá! Posso te ajudar com dúvidas sobre telemetria, frota ou sistemas do EiTruck. Por onde quer começar?"
        },
        {
            "human": "Me conta uma piada.",
            "ai": "Consigo ajudar apenas com informações técnicas do EiTruck. Quer saber sobre telemetria, monitoramento de frota ou sistemas de manutenção?"
        },
        {
            "human": "Como funciona o bloqueio remoto do veículo?",
            "ai": "ROUTE=\nPERGUNTA_ORIGINAL=Como funciona o bloqueio remoto do veículo?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
        },
        {
            "human": "Quero informações sobre sensores.",
            "ai": "Você quer detalhes sobre sensores para telemetria veicular ou sensores para integração industrial?"
        },
        {
            "human": "Quando a próxima manutenção da frota está agendada?",
            "ai": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Quando a próxima manutenção da frota está agendada?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
        },
    ]
    
    fewshots_roteador = FewShotChatMessagePromptTemplate(
        examples=shots_roteador,
        example_prompt=prompt_roteador,
    )

    prompt_roteador = ChatPromptTemplate.from_messages([
        prompt_roteador,                          # system prompt
        fewshots_roteador,                               # Shots human/ai 
        MessagesPlaceholder("chat_history"),    # memória
        ("human", "{input}"),                  # user prompt
        MessagesPlaceholder("agent_scratchpad") # acesso ao tools
    ])
    prompt_roteador = prompt_roteador.partial(today=today.isoformat(),time=hour.isoformat())
    agent_roteador = create_tool_calling_agent(llm_fast,prompt_roteador)
    agent_executor_roteador = AgentExecutor(agent=agent_roteador,verbose=False)
    chain_roteador = RunnableWithMessageHistory(
        agent_executor_roteador,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    return chain_roteador


def gemini_resp(user_id, session_id, question) -> str:
    session=f"{user_id}_{session_id}"
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.7, top_p=0.95, google_api_key=chave_api
    )
    
    with open("./app/ai/text/prompt_juiz.txt", "r", encoding="utf-8") as f:
        prompt_gemini_text = f.read()  

    system_prompt = (
        "system",
        prompt_gemini_text
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ]
    )
    shots = []
    fewshots = FewShotChatMessagePromptTemplate(examples=shots, example_prompt=prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            fewshots,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    base_chain = prompt | llm | StrOutputParser()
    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    if question.lower() in ("sair", "end", "fim", "tchau", "bye"):
        return "Encerrando o chat."
    try:
        resposta = chain.invoke(
            {"input": question}, config={"configurable": {"session_id": session}}
        )
        return resposta
    except Exception as e:
        return f"Não foi possível responder: {e}"




# Verificar Resposta
def juiz_resposta(question: str, answer: str,user_id: int,session_id: int) -> str:
    session=f"{user_id}_{session_id}"
    user = "Pergunta: "+question+"\nResposta: "+answer
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.5, google_api_key=chave_api
    )
    with open("./app/ai/text/prompt_juiz.txt", "r", encoding="utf-8") as f:
        prompt_juiz_text = f.read()
    system_prompt = (
        "system",
        prompt_juiz_text
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ]
    )
    shots = []
    fewshots = FewShotChatMessagePromptTemplate(examples=shots, example_prompt=prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            fewshots,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    base_chain = prompt | juiz | StrOutputParser()
    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    try:
        resposta_juiz = chain.invoke(
            {"input": user}, config={"configurable": {"session_id": session}}
        )
        resposta_juiz = resposta_juiz.strip()
        if resposta_juiz.startswith("```json"):
            resposta_juiz = resposta_juiz[len("```json"):].rstrip("```").strip()
        return resposta_juiz
    
    except Exception as e:
        return f"Não foi possível responder: {e}"