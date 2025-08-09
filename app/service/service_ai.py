from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta, gemini_resp
from app.ai.hist_chat import verifica_historico, insere_resposta
from datetime import datetime

user_id = 'Teste'

# Service
def question_for_gemini(question: str) -> dict:
    if verifica_pergunta(question) == "SIM":
        return\
            {
                "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
            }
    # Obtém resposta do RAG
    resposta_rag:str = rag_responder(question)
    
    # Verifica perguntas e respostas anteriores
    hist=verifica_historico(user_id)
    
    # Aciona o juiz com base na pergunta, resposta e histórico
    judgment:str = juiz_resposta(question, resposta_rag, hist)
        
    insere_resposta(judgment,hist,user_id)
    
    return \
        {   "timestamp": datetime.now().isoformat(),
            "content":{
                "rag":{
                    "judgment": judgment
                },
                "question": question,
            }
        }
