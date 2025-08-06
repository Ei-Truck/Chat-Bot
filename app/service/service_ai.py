from app.ai.ai_model import verifica_pergunta, rag_responder, juiz_resposta
from app.ai.histChat import verifica_historico, insere_resposta
from datetime import datetime

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
    hist=verifica_historico()
    
    # Aciona o juiz com base na pergunta, resposta e histórico
    judgment:str = juiz_resposta(question, resposta_rag, hist)
    
    insere_resposta(judgment,hist)
    
    return \
        {   "timestamp": datetime.now().isoformat(),
            "content":{
                "rag":{
                    "rag_answer": resposta_rag,
                    "judgment": judgment
                },
                "question": question,
            }
        }
