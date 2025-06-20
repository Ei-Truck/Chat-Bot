from modelo import verifica_pergunta, responder_pergunta, rag_responder, juiz_resposta
# Service

def question_for_gemini(question: str) -> dict:
    if verifica_pergunta(question) == "SIM":
        return\
            {
                "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
            }

    resposta:str = responder_pergunta(question)

    juiz_resposta_texto:str = juiz_resposta(question, resposta)

    return \
        {
            "question": question,
            "answer": resposta,
            "judgment": juiz_resposta_texto
        }


