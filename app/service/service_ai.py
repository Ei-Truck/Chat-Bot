from app.ai.ai_manager import models_management
from datetime import datetime


# Service
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:

    # Chama o gerenciador de modelos para obter a resposta final
    final_answer = models_management(id_user, id_session, question)

    # Retorna a resposta com timestamp
    return {
        "timestamp": datetime.now().isoformat(),
        "content": {
            "answer": final_answer,
            "question": question,
        },
    }
