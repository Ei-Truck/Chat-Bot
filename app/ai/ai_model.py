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
today_local = datetime.now(TZ).date()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")
mongo_host = os.getenv("CONNSTRING")


def get_llm_fast() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=chave_api,
    )


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.95,
        google_api_key=chave_api,
    )


# Verificar pergunta
def verifica_pergunta(pergunta: str) -> str:
    llm_fast = get_llm_fast()
    prompt_avaliacao = (
        "Você é um assistente que verifica se um texto contém linguagem ofensiva, discurso de ódio, "
        "calúnia ou difamação. Responda 'SIM' se contiver e 'NÃO' caso contrário. "
        "Seja estrito na sua avaliação."
    )
    resposta_llm = llm_fast.invoke(
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
    llm_fast = get_llm_fast()
    with open("./app/ai/text/prompt_roteador.txt", "r", encoding="utf-8") as f:
        prompt_roteador_text = f.read()

    system_prompt_roteador = ("system", prompt_roteador_text)

    prompt_roteador = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    shots_roteador = [
        {
            "input": "Oi, tudo bem?",
            "output": (
                "Olá! Sou o EiTruck.AI. Posso te ajudar com dúvidas sobre telemetria, frota ou sistemas do EiTruck. "
                "Sobre qual desses temas você quer falar?"
            ),
        },
        {
            "input": "Me conta uma piada.",
            "output": (
                "Eu só respondo a dúvidas técnicas sobre o EiTruck. "
                "Quer falar sobre telemetria, monitoramento de frota ou manutenção?"
            ),
        },
        {
            "input": "Como funciona o bloqueio remoto do veículo?",
            "output": (
                "ROUTE=automobilistica\n"
                "PERGUNTA_ORIGINAL=Como funciona o bloqueio remoto do veículo?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Quero informações sobre telemetria.",
            "output": (
                "ROUTE=faq\n"
                "PERGUNTA_ORIGINAL=Quero informações sobre telemetria.\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "O que é telemetria?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=O que é telemetria?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
    ]

    fewshots_roteador = FewShotChatMessagePromptTemplate(
        examples=shots_roteador, example_prompt=prompt_roteador
    )

    prompt_roteador = ChatPromptTemplate.from_messages(
        [
            system_prompt_roteador,
            fewshots_roteador,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt_roteador = prompt_roteador.partial(today_local=today_local.isoformat())
    chain_roteador = prompt_roteador | llm_fast | StrOutputParser()

    chain_roteador = RunnableWithMessageHistory(
        chain_roteador,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_roteador


# Especialista em automobilística
def especialista_auto(user_id, session_id) -> RunnableWithMessageHistory:
    llm = get_llm()
    with open(
        "./app/ai/text/prompt_especialista_automobilistica.txt", "r", encoding="utf-8"
    ) as f:
        prompt_especialista_text = f.read()

    system_prompt_especialista = ("system", prompt_especialista_text)

    prompt_especialista = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    shots_especialista = [
        {
            "input": (
                "ROUTE=automobilistica\nPERGUNTA_ORIGINAL=Qual é a principal função da telemetria?\n"
                "PERSONA={PERSONA_SISTEMA}\nCLARIFY="
            ),
            "output": (
                """{
                    "dominio": "automobilistica",
                    "resposta": "A telemetria é utilizada em diversas áreas, incluindo: "
                        "- Veículos (monitoramento de frota) "
                        "- Medicina (monitoramento de pacientes) "
                        "- Indústria (manutenção preditiva) "
                        "- Energia (monitoramento de redes elétricas) "
                        "- Agricultura (sensores em plantações) "
                        "- Esportes (dados de desempenho de atletas) "
                        "- Aviação (sistemas de voo e caixa preta) "
                        "- Defesa (monitoramento de drones e equipamentos remotos) "
                        "- Meteorologia (sensores climáticos remotos) "
                        "- Smart Cities (monitoramento de trânsito, iluminação e resíduos)"
            """
            ),
        }
    ]

    fewshots_especialista = FewShotChatMessagePromptTemplate(
        examples=shots_especialista, example_prompt=prompt_especialista
    )

    prompt_especialista = ChatPromptTemplate.from_messages(
        [
            system_prompt_especialista,
            fewshots_especialista,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt_especialista = prompt_especialista.partial(
        today_local=today_local.isoformat()
    )
    chain_auto = prompt_especialista | llm | StrOutputParser()

    chain_auto = RunnableWithMessageHistory(
        chain_auto,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_auto


# Juiz de respostas
def juiz_resposta(user_id: int, session_id: int) -> RunnableWithMessageHistory:
    juiz = get_llm()

    with open("./app/ai/text/prompt_juiz.txt", "r", encoding="utf-8") as f:
        prompt_juiz_text = f.read()

    system_prompt = ("system", prompt_juiz_text)

    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    base_chain = prompt | juiz | StrOutputParser()

    chain_juiz = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_juiz


# Especialista em perguntas gerais
def gemini_resp(user_id, session_id) -> RunnableWithMessageHistory:
    llm = get_llm()
    with open("./app/ai/text/prompt_gemini.txt", "r", encoding="utf-8") as f:
        prompt_gemini_text = f.read()
    system_prompt = ("system", prompt_gemini_text)
    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ]
    )
    shots = [
        {
            "input": "O que é telemetria?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=O que é telemetria?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Como o EiTruck coleta os dados dos veículos?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=Como o EiTruck coleta os dados dos veículos?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Quais são os principais serviços oferecidos pela EiTruck?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=Quais são os principais serviços oferecidos pela EiTruck?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "O sistema do EiTruck precisa de internet para funcionar?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=O sistema do EiTruck precisa de internet para funcionar?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Como posso acessar os relatórios de telemetria?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=Como posso acessar os relatórios de telemetria?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "O que diferencia o EiTruck de outros sistemas de gestão de frota?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=O que diferencia o EiTruck de outros sistemas de gestão de frota?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Posso integrar o EiTruck com outros sistemas da empresa?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=Posso integrar o EiTruck com outros sistemas da empresa?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Os dados de telemetria são armazenados por quanto tempo?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=Os dados de telemetria são armazenados por quanto tempo?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "Como funciona o suporte técnico do EiTruck?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=Como funciona o suporte técnico do EiTruck?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
        {
            "input": "O que é análise de comportamento do motorista?",
            "output": (
                "ROUTE=outros\n"
                "PERGUNTA_ORIGINAL=O que é análise de comportamento do motorista?\n"
                "PERSONA={PERSONA_SISTEMA}\n"
                "CLARIFY="
            ),
        },
    ]
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
    chain_gemini = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_gemini


# Agente Orquestrador
def orquestrador_resp(user_id: int, session_id: int) -> RunnableWithMessageHistory:
    llm = get_llm()

    with open("./app/ai/text/prompt_orquestrador.txt", "r", encoding="utf-8") as f:
        system_orquestrador_prompt = f.read()

    system_orquestrador_prompt = ("system", system_orquestrador_prompt)

    shots_orquestrador = [
        {
            "input": (
                "ESPECIALISTA_JSON:{{'dominio':'automobilistica','resposta':'Você gastou R$ 842,75 com comida',"
                "'recomendacao':'Quer detalhar por estabelecimento?'}}"
            ),
            "output": (
                "Você gastou R$ 842,75 com comida.\n*Recomendação*:\nQuer detalhar por estabelecimento?"
            ),
        }
    ]

    example_prompt_orquestrador = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    fewshots_orquestrador = FewShotChatMessagePromptTemplate(
        examples=shots_orquestrador, example_prompt=example_prompt_orquestrador
    )

    prompt_orquestrador = ChatPromptTemplate.from_messages(
        [
            system_orquestrador_prompt,
            fewshots_orquestrador,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    base_chain = prompt_orquestrador | llm | StrOutputParser()

    chain_orquestrador = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_orquestrador
