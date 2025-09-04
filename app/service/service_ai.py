from app.ai.ai_model import verifica_pergunta, juiz_resposta, gemini_resp
from app.ai.ai_rag import embedding_files, search_embedding
from datetime import datetime
import json


# Service
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:
    
    # Embeddando o arquivo txt de FAQ
    embedding_files()
    
    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }
    
    encontrado = search_embedding(question)
    score = encontrado[0]
    
    if float(score[0]) <= 0.6:    
        answer = gemini_resp(id_user,id_session,question)
    else:
        answer = encontrado[1]

    judgment: str = juiz_resposta(question, answer)

    juiz = json.loads(judgment)
    status = juiz["status"]

    if status == "Aprovado":
        final_answer = juiz["answer"]
    elif status == "Reprovado":
        final_answer = juiz["judgmentAnswer"]


    return {
        "timestamp": datetime.now().isoformat(),
        "content": {
            "answer": final_answer,
            "question": question,
        },
    }
