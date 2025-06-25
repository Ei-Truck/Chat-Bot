from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta
import json
from datetime import datetime
# Service

def question_for_gemini(question: str) -> dict:
    if verifica_pergunta(question) == "SIM":
        return\
            {
                "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
            }

    resposta_rag:str = rag_responder(question)
    judgment:str = juiz_resposta(question, resposta_rag)
    # Converter a string JSON em dicionário Python
    dados = json.loads(judgment)
    # Acessar o atributo status
    if dados["status"] == "Aprovado":
        response = dados["answer"]
    else:
        response = dados["judgmentAnswer"]
    status = dados["status"]

    return \
        {   "timestamp": datetime.now().isoformat(),
            "content":{
                "status": status,
                "answer": response,
                "question": question,
            }
        }


