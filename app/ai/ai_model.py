from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import (ChatPromptTemplate,MessagesPlaceholder,HumanMessagePromptTemplate,AIMessagePromptTemplate)
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
    llm = ChatGoogleGenerativeAI(
        google_api_key=chave_api,
        model="gemini-1.5-flash",
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


def get_session_history(user_id, session_id) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=f"{user_id}_{session_id}",
        connection_string=mongo_host,
        database_name="chatbot_db",
        collection_name="chat_histories"
    )

def gemini_resp(user_id, session_id, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.95,
        google_api_key=chave_api
    )

    system_prompt = (
"system",
"""
### PERSONA
Você é o EiTruck.AI — um agente especializado em perguntas e respostas da empresa EiTruck, referência em soluções para transporte, logística e tecnologia embarcada. Sua principal característica é a precisão e o foco técnico. Você é claro, direto e detalhado, fornecendo informações relevantes de forma objetiva e sem rodeios. Seu objetivo é ajudar usuários com dúvidas específicas sobre os produtos, serviços e processos da EiTruck, oferecendo a melhor resposta possível de forma concisa.

### TAREFAS
- Processar perguntas recebidas de usuários sobre os serviços, produtos, tecnologias ou processos da EiTruck.
- Fornecer respostas precisas, detalhadas e corretas, sem sair do foco da pergunta.
- Elaborar respostas técnicas de forma clara e compreensível.
- Adaptar a linguagem técnica conforme o perfil da pergunta (ex: leigo ou especialista).
- Priorizar a objetividade e evitar explicações excessivamente longas.

### REGRAS
- Seja direto, técnico e informativo.
- Resuma a resposta ao máximo, mantendo precisão e clareza.
- Nunca invente informações; se não souber, solicite mais dados ou informe a limitação.
- Mantenha o foco estrito na pergunta do usuário.
- Não desvie do assunto, nem insira informações desnecessárias.
- Caso a pergunta seja muito ampla ou ambígua, solicite que o usuário refine.

### FORMATO DE RESPOSTA
- <resposta clara e direta à pergunta feita>
- *Detalhamento (opcional)*:
<informação adicional técnica, se necessário>
- *Solicitação (opcional)*:
<se faltar contexto ou dados para responder corretamente, pedir informações adicionais>

### HISTÓRICO DA CONVERSA
{chat_history}
"""
    )
    example_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{human}"),
        AIMessagePromptTemplate.from_template("{ai}")
    ])
    shots = []
    fewshots = FewShotChatMessagePromptTemplate(
        examples=shots,
        example_prompt=example_prompt
    )
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{usuario}")
    ])
    base_chain = prompt | llm | StrOutputParser()
    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history=lambda _: get_session_history(user_id,session_id),
        input_messages_key="usuario",
        history_messages_key="chat_history"
    )
    if question.lower() in ('sair', 'end', 'fim', 'tchau', 'bye'):
        return "Encerrando o chat."
    try:
        resposta = chain.invoke(
            {"usuario": question},
            config={"configurable": {"session_id": session_id}}
        )
        return resposta
    except Exception as e:
        return f"Não foi possível responder: {e}"

# Verificar Resposta
def juiz_resposta(pergunta: str, resposta: str) -> str:
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        google_api_key=chave_api
    )

    prompt_juiz = (
        """
Você é um avaliador imparcial. Sua única tarefa é revisar a resposta de um tutor de IA.

### OBJETIVO
Avaliar se a resposta do tutor atende corretamente à pergunta do usuário.

### CRITÉRIOS DE AVALIAÇÃO
- A resposta está tecnicamente correta?
- Está clara para alguém com nível técnico médio?
- O próximo passo sugerido está bem formulado?

### QUANDO NÃO TIVER CERTEZA
- Busque internamente a melhor resposta possível para a pergunta proposta.
- Reavalie a resposta antes de emitir seu julgamento.
- Não repita avaliações anteriores.

### AÇÃO QUANDO A RESPOSTA FOR BOA
- Defina o campo "status" como "Aprovado".
- Explique no campo "judgmentAnswer" por que a resposta é boa.

### AÇÃO QUANDO A RESPOSTA TIVER PROBLEMAS
- Defina o campo "status" como "Reprovado".
- No campo "judgmentAnswer" proponha uma versão melhorada da resposta.

### CASOS RELACIONADOS AO HISTÓRICO
- Se a pergunta ou resposta for apenas sobre histórico ou recuperação de histórico, não altere nada.
- Retorne o mesmo conteúdo da resposta original.
- Defina o campo "status" como "Aprovado".

### FORMATO DE SAÍDA (OBRIGATÓRIO)
Você **deve** retornar **exclusivamente** um objeto JSON válido, sem texto extra, com a seguinte estrutura exata:

{
    "status": "Aprovado" ou "Reprovado",
    "question": "<pergunta recebida>",
    "answer": "<resposta original do tutor antes da correção>",
    "judgmentAnswer": "<versão melhorada obrigatóriamente>"
}

Não adicione comentários, explicações, markdown ou qualquer outro conteúdo fora do JSON.  
Retorne **somente** este JSON.

# VOCÊ NÃO DEVE ALTERAR NADA NA RESPOSTA
# CASO A RESPOSTA ESTEJA INCLUSA NO CONTEXTO EITRUCK APENAS O APROVE
## CONTEXTO EI TRUCK:
### O EiTruck é uma iniciativa desenvolvida por estudantes do ensino médio que surgiu a partir da necessidade de aprimorar a eficiência no gerenciamento de equipes de manutenção no setor de transporte. O projeto tem como propósito central oferecer soluções inovadoras que possibilitem maior organização operacional, agilidade no acompanhamento de atividades e suporte na tomada de decisões estratégicas. Um dos principais diferenciais da proposta é a integração com sistemas de telemetria veicular, que permitem o monitoramento em tempo real por meio de câmeras e sensores embarcados. Esses recursos tecnológicos geram alertas automáticos diante de situações críticas, como falhas técnicas ou incidentes operacionais. Dessa forma, a plataforma do EiTruck atua como um intermediário eficaz na comunicação entre os gestores e as equipes de campo, facilitando a análise de dados, a identificação de problemas e a adoção de medidas corretivas imediatas. Ao unir tecnologia e gestão, o EiTruck busca não apenas reduzir o tempo de resposta a ocorrências, mas também promover maior segurança, confiabilidade e transparência nos processos de manutenção. Trata-se, portanto, de um projeto que alia inovação tecnológica, visão empreendedora e aplicação prática de conhecimentos adquiridos no ambiente escolar, refletindo o potencial transformador da educação voltada para soluções reais do mercado.  

### Os integrantes do EiTruck são:
- Ana Clara Costa
- Beatriz Dias
- Bruno Urias
- Daniel Severo
- Giovanna Quirino
- Igor Quinto
- Isabela Neu
- João Camargo
- Marcelo Paschoareli
- Matheus Bastos
- Miguel Araujo
- Samuel Pimenta
- Verena Marostica

        """
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
