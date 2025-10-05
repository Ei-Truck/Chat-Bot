from app.ai.ai_model import (
    verifica_pergunta,
    roteador_eitruck,
    especialista_auto,
    gemini_resp,
    juiz_resposta,
    orquestrador_resp,
)
from app.ai.ai_rag import embedding_files, search_embedding
import json


def models_management(user_id, session_id, question):
    embedding_files()

    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }

    encontrado = search_embedding(question)
    score = encontrado[0]

    if float(score[0]) <= 0.6:
        session = f"{user_id}_{session_id}"

        resposta_roteador = roteador_eitruck(user_id, session_id).invoke(
            {"input": question},
            config={"configurable": {"session_id": session}},
        )

        if "ROUTE=" not in resposta_roteador:
            return resposta_roteador

        elif "ROUTE=agenda" in resposta_roteador:
            resposta = especialista_auto(user_id, session_id).invoke(
                {"input": resposta_roteador},
                config={"configurable": {"session_id": session}},
            )

        elif "ROUTE=financeiro" in resposta_roteador:
            resposta = gemini_resp(user_id, session_id).invoke(
                {"input": resposta_roteador},
                config={"configurable": {"session_id": session}},
            )

        if not isinstance(resposta, dict):
            resposta = json.loads(resposta)

        resposta = resposta.get("output", resposta)

    else:
        resposta = encontrado[1]

    resposta = juiz_resposta(user_id, session_id).invoke(
        {"input": resposta},
        config={"configurable": {"session_id": session}},
    )

    if not isinstance(resposta, dict):
        resposta = json.loads(resposta)

    resposta = resposta.get("output", resposta)

    resposta = orquestrador_resp(user_id, session_id).invoke(
        {"input": resposta},
        config={"configurable": {"session_id": session}},
    )

    if isinstance(resposta, str):
        try:
            resposta = json.loads(resposta)
            resposta = resposta.get("output", resposta)
        except json.JSONDecodeError:
            pass

    return resposta
