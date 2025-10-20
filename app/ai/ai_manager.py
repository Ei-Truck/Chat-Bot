from app.ai.ai_model import (
    verifica_pergunta,
    roteador_eitruck,
    especialista_auto,
    gemini_resp,
    juiz_resposta,
    orquestrador_resp,
    especialista_faq,
)
from app.ai.ai_rag import embedding_files
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
        resposta = especialista_faq().invoke(
            {"input": original_question},
            config={"configurable": {"session_id": session}},
        )
        return resposta

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
        if "```json" in resposta:
            resposta = resposta.split("```json")[-1].strip()
        if "```" in resposta:
            resposta = resposta.split("```")[0].strip()
        resposta = json.loads(resposta)
        resposta = resposta.get("output", resposta)
        return _finalizar_resposta(user_id, session_id, resposta)

    return {
        "error": "Não foi possível processar a pergunta no momento. Tente novamente mais tarde."
    }


def _finalizar_resposta(user_id, session_id, resposta) -> str:
    session = f"{user_id}_{session_id}"

    if isinstance(resposta, dict):
        resposta_str = json.dumps(resposta, ensure_ascii=False)
    else:
        resposta_str = str(resposta)

    resposta = juiz_resposta(user_id, session_id).invoke(
        {"input": resposta_str},
        config={"configurable": {"session_id": session}},
    )

    if "```json" in resposta:
        resposta = resposta.split("```json")[-1].strip()
    if "```" in resposta:
        resposta = resposta.split("```")[0].strip()

    try:
        resposta_json = json.loads(resposta)
        resposta = resposta_json.get("output", resposta_json)
    except json.JSONDecodeError:
        pass

    if isinstance(resposta, dict):
        resposta_str = json.dumps(resposta, ensure_ascii=False)
    else:
        resposta_str = str(resposta)

    resposta = orquestrador_resp(user_id, session_id).invoke(
        {"input": resposta_str},
        config={"configurable": {"session_id": session}},
    )

    if isinstance(resposta, str):
        try:
            resposta_json = json.loads(resposta)
            resposta = resposta_json.get("output", resposta_json)
        except json.JSONDecodeError:
            pass

    return resposta
