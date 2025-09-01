from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta, gemini_resp
from app.ai.embedding import verifica_embedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.ai.histChat import ChatHistory, salvar_historico, get_chat_hist, get_session_history
from datetime import datetime
from dotenv import load_dotenv
import json
import os

load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")

hist = ChatHistory()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,credentials=chave_api)

# Service
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:
    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }

    memory = ConversationBufferMemory(
        chat_memory=get_chat_hist(id_user, id_session),
        return_messages=True
    )

    conversation = RunnableWithMessageHistory(
        runnable=llm,
        get_session_history=lambda session_id=id_session: get_session_history(session_id)
    )
    contexto = hist.search_history(id_user, id_session, question)
    contexto_texto = "Contexto de conversas anteriores:\n" if contexto else ""
    if contexto:
        for c in contexto:
            if not c:
                continue
            if isinstance(c, dict):
                c_dict = c
            elif isinstance(c, str):
                try:
                    c_dict = json.loads(c)
                except json.JSONDecodeError:
                    c_dict = {"user": id_user, "mensagem": c}
            else:
                continue
            contexto_texto += f"{c_dict['user']}: {c_dict['mensagem']}\n"

    prompt = f"{contexto_texto}\nUsuário: {question}"

    resposta = rag_responder(question)
    if resposta is not None:
        resposta_texto, resposta_score = resposta[0]

    encontrado = verifica_embedding(question)

    if encontrado:
        if resposta_score < 0.5:
            resposta_texto = gemini_resp(prompt)

        resposta_chain = conversation.invoke({"input": prompt}).content
        resposta_completa = f"{resposta_texto}\n\n[Histórico de conversa]: {resposta_chain}"

        try:
            judgment = juiz_resposta(prompt, resposta_completa)
            juiz = json.loads(judgment)
            status = juiz.get("status", "Aprovado")

            if status == "Aprovado":
                final_answer = juiz.get("answer", resposta_completa)
            else:
                final_answer = juiz.get("judgmentAnswer", resposta_completa)
        except Exception:
            final_answer = resposta_completa

        salvar_historico(id_user, id_session, question, final_answer)
        conversation.memory.save_context({"question": question}, {"answer": final_answer})

    else:
        final_answer = encontrado

    return {
        "timestamp": datetime.now().isoformat(),
        "content": {
            "answer": final_answer,
            "question": question,
        },
    }
