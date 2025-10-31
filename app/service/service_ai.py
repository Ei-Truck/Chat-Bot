from app.ai.ai_manager import models_management
from datetime import datetime


# Service que processa a pergunta do usuário e retorna a resposta final
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:
    """
    Recebe a pergunta do usuário, ID do usuário e ID da sessão,
    e retorna a resposta processada pelo gerenciador de modelos AI.
    """

    # Chama o gerenciador de modelos que decide qual LLM/fluxo usar
    final_answer = models_management(id_user, id_session, question)

    # Retorna a resposta estruturada em JSON com timestamp e conteúdo
    return {
        "timestamp": datetime.now().isoformat(),  # Data e hora da resposta
        "content": {
            "answer": final_answer,  # Resposta gerada pelo AI
            "question": question,  # Pergunta original do usuário
        },
    }
