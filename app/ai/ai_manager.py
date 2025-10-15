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


def models_management(user_id, session_id, question) -> str:
    embedding_files()
    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }
    return _processar_pergunta(user_id, session_id, question)

def _processar_pergunta(user_id, session_id, question) -> str:
    original_question = question    
    session = f"{user_id}_{session_id}"
    resposta_roteador = roteador_eitruck(user_id, session_id).invoke(
        {"input": question},
        config={"configurable": {"session_id": session}},
    )

    if "ROUTE=" not in resposta_roteador:
        return resposta_roteador
    
    if "ROUTE=faq" in resposta_roteador:
        encontrado = search_embedding(original_question, top_k=1)
        score = float(encontrado[0][0])
        if score <= 0.8:
            resposta = gemini_resp(user_id, session_id).invoke(
            {"input": resposta_roteador},
            config={"configurable": {"session_id": session}},
        )
        else:
            resposta = encontrado[0][1]
            return _finalizar_resposta(user_id, session_id, resposta)

    if "ROUTE=automobilistica" in resposta_roteador:
        resposta = especialista_auto(user_id, session_id).invoke(
            {"input": resposta_roteador},
            config={"configurable": {"session_id": session}},
        )
    elif "ROUTE=outros" in resposta_roteador:
        resposta = gemini_resp(user_id, session_id).invoke(
            {"input": resposta_roteador},
            config={"configurable": {"session_id": session}},
        )
    if resposta:
        if not isinstance(resposta, dict):
            resposta = json.loads(resposta)
        return resposta.get("output", resposta)
    return {
            "error": "Não foi possível processar a pergunta no momento. Tente novamente mais tarde."
        }

def _finalizar_resposta(user_id, session_id, resposta) -> str:
    session = f"{user_id}_{session_id}"

    resposta = juiz_resposta(user_id, session_id).invoke(
        {"input": resposta},
        config={"configurable": {"session_id": session}},
    )
    if "```json" in resposta:
        resposta = resposta.split("```json")[-1].strip()
    if "```" in resposta:
        resposta = resposta.split("```")[0].strip()

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
