from app.ai.ai_model import verifica_pergunta, juiz_resposta, gemini_resp
from app.ai.ai_rag import embedding_store_documents_from_files, embedding_store_documents_from_mongo, search_embedding
from datetime import datetime
import json


# Service
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:

    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }
    
    answer = gemini_resp(id_user,id_session,question)

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
