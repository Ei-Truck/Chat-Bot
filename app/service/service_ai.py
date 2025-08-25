from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta, gemini_resp
from app.ai.embedding import historico_gemini,verifica_embedding
from app.ai.histChat import ChatHistory
from datetime import datetime
import json

# Instanciando histórico
hist = ChatHistory()

# Service
def question_for_gemini(question: str) -> dict:
    user_id = "Teste"

    # 🚨 Verificação de conteúdo ofensivo
    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }

    # 🔹 Salva a pergunta no histórico
    hist.armazenar_mensagem("user", question)

    # 🔹 Busca contexto no histórico
    contexto = hist.search_history(question)
    contexto_texto = ""
    if contexto != 0:
        contexto_texto = "Contexto de conversas anteriores:\n"
        for c in contexto:
            contexto_texto += f"{c['user']}: {c['mensage']}\n"

    # 🔹 monta prompt com contexto + pergunta
    prompt = f"{contexto_texto}\nUsuário: {question}\nBot:"

    # 🔹 Obtém resposta do RAG
    resposta = rag_responder(user_id, question)
    resposta_texto, resposta_score = resposta[0]

    # 🔹 Verifica se já existe embedding correspondente
    encontrado = verifica_embedding(user_id, question, resposta_texto)

    if encontrado is None:
        if resposta_score < 0.5:
            # Usa Gemini com suporte do histórico
            resposta_texto = gemini_resp(prompt)
        else:
            resposta_texto = resposta_texto

        # Juiz de resposta (se existir lógica de validação extra)
        judgment: str = juiz_resposta(prompt, resposta_texto)
        
        juiz = json.loads(judgment)
        status = juiz["status"]

        if status == "Aprovado":
            final_answer = juiz["answer"]
        elif status == "Reprovado":
            final_answer = juiz["judgmentAnswer"]
        
        historico_gemini(user_id,question,str(final_answer))
        hist.armazenar_mensagem("bot", str(final_answer))
    else:
        final_answer = encontrado
        

    return \
        {   "timestamp": datetime.now().isoformat(),
            "content":{
                "answer": final_answer,

                "question": question,
            }
        }