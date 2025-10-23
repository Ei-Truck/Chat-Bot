from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from app.ai.ai_rag import get_faq_context
from zoneinfo import ZoneInfo
from datetime import datetime
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import os
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Carrega variáveis de ambiente do arquivo .env (como chaves de API e conexões)
load_dotenv()

# Define o fuso horário de São Paulo e captura a data local atual
TZ = ZoneInfo("America/Sao_Paulo")
today_local = datetime.now(TZ).date()

# Recupera as credenciais e conexões do ambiente
chave_api = os.getenv("GEMINI_API_KEY")
mongo_host = os.getenv("CONNSTRING")


# Função para criar um modelo Gemini leve e rápido, usado em tarefas simples
def get_llm_fast() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=chave_api,
    )


# Função que retorna um modelo Gemini mais sofisticado, com criatividade moderada
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.95,
        google_api_key=chave_api,
    )


# Função que verifica se uma pergunta contém linguagem ofensiva
def verifica_pergunta(pergunta: str) -> str:
    llm_fast = get_llm_fast()
    prompt_avaliacao = (
        "Você é um assistente que verifica se um texto contém linguagem ofensiva, discurso de ódio, "
        "calúnia ou difamação. Responda 'SIM' se contiver e 'NÃO' caso contrário. "
        "Seja estrito na sua avaliação."
    )
    # O modelo analisa a pergunta e retorna se é apropriada ou não
    resposta_llm = llm_fast.invoke(
        [HumanMessage(content=prompt_avaliacao + "\n\nPergunta: " + pergunta)]
    )
    return resposta_llm.content.strip()


# Função que cria e gerencia o histórico de conversas do usuário no MongoDB
def get_session_history(user_id, session_id) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=f"{user_id}_{session_id}",
        connection_string=mongo_host,
        database_name="chatbot_db",
        collection_name="chat_histories",
    )


# Agente responsável por direcionar a conversa ao especialista correto
def roteador_eitruck(user_id, session_id) -> RunnableWithMessageHistory:
    llm_fast = get_llm_fast()
    # Lê o prompt base do roteador (define como ele deve pensar)
    with open("./app/ai/text/prompt_roteador.txt", "r", encoding="utf-8") as f:
        prompt_roteador_text = f.read()

    system_prompt_roteador = ("system", prompt_roteador_text)

    # Estrutura dos exemplos que o modelo usará para aprender o padrão de respostas
    prompt_roteador = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    # Exemplos práticos (few-shots) que ajudam o modelo a entender o estilo de resposta esperado
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

    # Cria um prompt few-shot a partir dos exemplos
    fewshots_roteador = FewShotChatMessagePromptTemplate(
        examples=shots_roteador, example_prompt=prompt_roteador
    )

    # Define a estrutura final do prompt com o histórico de conversa
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

    # Adiciona suporte a histórico de mensagens no MongoDB
    chain_roteador = RunnableWithMessageHistory(
        chain_roteador,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_roteador


# Agente especialista em temas automobilísticos
def especialista_auto(user_id, session_id) -> RunnableWithMessageHistory:
    llm = get_llm()
    with open(
        "./app/ai/text/prompt_especialista_automobilistica.txt", "r", encoding="utf-8"
    ) as f:
        prompt_especialista_text = f.read()

    system_prompt_especialista = ("system", prompt_especialista_text)

    # Define exemplos de entrada e saída esperados para o especialista
    prompt_especialista = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    # Exemplo prático de resposta do especialista
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

    # Monta o prompt completo e cria a cadeia com histórico
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


# Agente “juiz”, responsável por avaliar e escolher a melhor resposta entre especialistas
def juiz_resposta(user_id: int, session_id: int) -> RunnableWithMessageHistory:
    juiz = get_llm()

    with open("./app/ai/text/prompt_juiz.txt", "r", encoding="utf-8") as f:
        prompt_juiz_text = f.read()

    system_prompt = ("system", prompt_juiz_text)

    # Estrutura do prompt do juiz
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    base_chain = prompt | juiz | StrOutputParser()

    # Encapsula o juiz com histórico de mensagens
    chain_juiz = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_juiz


# Agente para perguntas gerais, respondidas de forma ampla e contextual
def gemini_resp(user_id, session_id) -> RunnableWithMessageHistory:
    llm = get_llm()
    with open("./app/ai/text/prompt_gemini.txt", "r", encoding="utf-8") as f:
        prompt_gemini_text = f.read()
    system_prompt = ("system", prompt_gemini_text)

    # Estrutura dos exemplos usados pelo modelo
    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    # Exemplos de perguntas gerais para guiar o modelo
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
        # (... demais exemplos seguem o mesmo formato)
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


# Agente orquestrador — combina e formata as respostas dos outros agentes
def orquestrador_resp(user_id: int, session_id: int) -> RunnableWithMessageHistory:
    llm = get_llm()

    with open("./app/ai/text/prompt_orquestrador.txt", "r", encoding="utf-8") as f:
        system_orquestrador_prompt = f.read()

    system_orquestrador_prompt = ("system", system_orquestrador_prompt)

    # Exemplo de como o orquestrador deve interpretar e formatar respostas
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


# Especialista em FAQ — busca respostas diretamente no contexto dos documentos (RAG)
def especialista_faq() -> RunnablePassthrough:

    llm_fast = get_llm_fast()

    with open("./app/ai/text/prompt_faq.txt", "r", encoding="utf-8") as f:
        prompt_faq_text = f.read()

    system_prompt_faq = ("system", prompt_faq_text)

    # Estrutura do prompt para o FAQ: combina a pergunta com o contexto encontrado
    prompt_faq = ChatPromptTemplate.from_messages(
        [
            system_prompt_faq,
            (
                "human",
                "Pergunta do usuário:\n{question}\n\n"
                "CONTEXTO (trechos do documento):\n{context}\n\n"
                "Responda com base APENAS no CONTEXTO.",
            ),
        ]
    )

    # Cria a cadeia que busca o contexto e o envia ao modelo
    faq_chain_core = (
        RunnablePassthrough.assign(
            question=itemgetter("input"),
            context=lambda x: get_faq_context(
                x["input"] if isinstance(x, dict) else x.page_content
            ),
        )
        | prompt_faq
        | llm_fast
        | StrOutputParser()
    )

    return faq_chain_core
