from app.ai.ai_model import verifica_pergunta, juiz_resposta, gemini_resp
from datetime import datetime
import json


# Service
def question_for_gemini(question: str, id_user: int) -> dict:
    user_id = str(id_user)

    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }
    
    resposta_texto = gemini_resp(question)

    judgment: str = juiz_resposta(question, resposta_texto)

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
