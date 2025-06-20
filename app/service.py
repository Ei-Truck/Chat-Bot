from ai.ai_model import verifica_pergunta, responder_pergunta, rag_responder, juiz_resposta
from datetime import datetime
# Service

def question_for_gemini(question: str) -> dict:
    if verifica_pergunta(question) == "SIM":
        return\
            {
                "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
            }

    resposta:str = responder_pergunta(question)
    resposta_rag:str = rag_responder(question)

    return \
        {   "timestamp": datetime.now().isoformat(),
            "content":{
                "rag":{
                    "rag_answer": resposta_rag,
                    "judgment": juiz_resposta(question, resposta_rag)
                },
                "gemini":{
                    "gemini_answer": resposta,
                    "judgment": juiz_resposta(question, resposta)
                },
                "question": question,
            }
        }


