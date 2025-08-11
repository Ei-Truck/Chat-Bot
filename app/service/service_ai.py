from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta, gemini_resp
from app.ai.hist_chat import verifica_historico, insere_resposta
from app.ai.embedding import historico_gemini
from datetime import datetime
import json


user_id = 'Teste'

# Service
def question_for_gemini(question: str) -> dict:
    if verifica_pergunta(question) == "SIM":
        return\
            {
                "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
            }
    # Obtém resposta do RAG
    resposta = rag_responder(question)
    print(resposta)
    if resposta[0][1] < 0.5:
        resposta:str = gemini_resp(question)
    elif resposta[0][1] >= 0.5:
        resposta = resposta[0][0]
    
    judgment:str = juiz_resposta(question, resposta)
        
    juiz = json.loads(judgment)
    status = juiz["status"]

    if status == "Aprovado":
        final_answer = juiz["answer"]
    elif status == "Reprovado":
        final_answer = juiz["judgmentAnswer"]
    
    historico_gemini(user_id,final_answer,question)


    return \
        {   "timestamp": datetime.now().isoformat(),
            "content":{
                "answer": final_answer,
                "question": question,
            }
        }
