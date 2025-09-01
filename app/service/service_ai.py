from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta, gemini_resp
from app.ai.embedding import verifica_embedding
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from app.ai.histChat import ChatHistory, salvar_historico,get_chat_hist
from datetime import datetime
import json

# Instanciando histórico
hist = ChatHistory()

model = genai.GenerativeModel("gemini-2.0-flash")
context = ConversationBufferMemory()


# Service
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:
    
    memory = ConversationBufferMemory(chat_memory=get_chat_hist(id_user,id_session),return_messages=True)
    conversation = ConversationChain(llm=model,memory=memory)

    conversation.run(question)
    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }

    contexto = hist.search_history(id_user,id_session,question)
    contexto_texto = ""
    if contexto:
        contexto_texto = "Contexto de conversas anteriores:\n"
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

    resposta = rag_responder(id_user, question)
    resposta_texto, resposta_score = resposta[0]

    encontrado = verifica_embedding(question)

    if encontrado is None:
        if resposta_score < 0.5:
            # Usa Gemini com suporte do histórico
            resposta_texto = gemini_resp(prompt)
        else:
            resposta_texto = resposta_texto

        judgment: str = juiz_resposta(prompt, resposta_texto)

        juiz = json.loads(judgment)
        status = juiz["status"]

        if status == "Aprovado":
            final_answer = juiz["answer"]
        elif status == "Reprovado":
            final_answer = juiz["judgmentAnswer"]
        
        salvar_historico(id_user,id_session,question,final_answer)
        context.save_context({"question":question},{"answer":final_answer})
        
    else:
        final_answer = encontrado

    return {
        "timestamp": datetime.now().isoformat(),
        "content": {
            "answer": final_answer,
            "question": question,
        },
    }
