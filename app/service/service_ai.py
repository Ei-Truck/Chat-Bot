from app.ai.ai_manager import models_management
from datetime import datetime
import json


# Service
def question_for_gemini(question: str, id_user: int, id_session: int) -> dict:

    final_answer = models_management(id_user,id_session,question)

    return {
        "timestamp": datetime.now().isoformat(),
        "content": {
            "answer": final_answer,
            "question": question,
        },
    }
