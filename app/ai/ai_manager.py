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


# Função principal que gerencia a pergunta do usuário
def models_management(user_id, session_id, question) -> str:
    # Atualiza embeddings dos arquivos, garantindo que a base de conhecimento esteja pronta
    embedding_files()

    # Verifica se a pergunta contém conteúdo impróprio
    if verifica_pergunta(question) == "SIM":
        return {
            "error": "Pergunta contém linguagem ofensiva, discurso de ódio, calúnia ou difamação."
        }

    # Processa a pergunta normalmente se estiver adequada
    return _processar_pergunta(user_id, session_id, question)


# Função interna que processa a pergunta com base na rota definida pelo roteador
def _processar_pergunta(user_id, session_id, question) -> str:
    original_question = question
    session = f"{user_id}_{session_id}"

    # Chama o roteador para decidir qual especialista deve responder
    resposta_roteador = roteador_eitruck(user_id, session_id).invoke(
        {"input": question},
        config={"configurable": {"session_id": session}},
    )

    # Se o roteador não indicar nenhuma rota, retorna a resposta diretamente
    if "ROUTE=" not in resposta_roteador:
        return resposta_roteador

    # Caso a rota seja FAQ, utiliza o especialista de FAQ
    if "ROUTE=faq" in resposta_roteador:
        resposta = especialista_faq().invoke(
            {"input": original_question},
            config={"configurable": {"session_id": session}},
        )
        return resposta

    # Caso a rota seja automobilística, chama o especialista de automóveis
    if "ROUTE=automobilistica" in resposta_roteador:
        resposta = especialista_auto(user_id, session_id).invoke(
            {"input": resposta_roteador},
            config={"configurable": {"session_id": session}},
        )
    # Caso a rota seja "outros", utiliza o agente Gemini para respostas gerais
    elif "ROUTE=outros" in resposta_roteador:
        resposta = gemini_resp(user_id, session_id).invoke(
            {"input": resposta_roteador},
            config={"configurable": {"session_id": session}},
        )

    # Limpeza de formatação e conversão para JSON, se aplicável
    if resposta:
        if "```json" in resposta:
            resposta = resposta.split("```json")[-1].strip()
        if "```" in resposta:
            resposta = resposta.split("```")[0].strip()
        resposta = json.loads(resposta)
        resposta = resposta.get("output", resposta)

        # Passa a resposta pelo fluxo final de validação e orquestração
        return _finalizar_resposta(user_id, session_id, resposta)

    # Retorna erro caso nenhum agente consiga processar a pergunta
    return {
        "error": "Não foi possível processar a pergunta no momento. Tente novamente mais tarde."
    }


# Função interna que finaliza a resposta, passando pelos agentes juiz e orquestrador
def _finalizar_resposta(user_id, session_id, resposta) -> str:
    session = f"{user_id}_{session_id}"

    # Converte a resposta para string JSON, se necessário
    if isinstance(resposta, dict):
        resposta_str = json.dumps(resposta, ensure_ascii=False)
    else:
        resposta_str = str(resposta)

    # Passa a resposta pelo juiz para validação e ajuste
    resposta = juiz_resposta(user_id, session_id).invoke(
        {"input": resposta_str},
        config={"configurable": {"session_id": session}},
    )

    # Limpa marcações de código e formatação
    if "```json" in resposta:
        resposta = resposta.split("```json")[-1].strip()
    if "```" in resposta:
        resposta = resposta.split("```")[0].strip()

    # Tenta converter a resposta para JSON e extrair o campo 'output'
    try:
        resposta_json = json.loads(resposta)
        resposta = resposta_json.get("output", resposta_json)
    except json.JSONDecodeError:
        pass

    # Garante que a resposta esteja em formato de string
    if isinstance(resposta, dict):
        resposta_str = json.dumps(resposta, ensure_ascii=False)
    else:
        resposta_str = str(resposta)

    # Passa a resposta final pelo orquestrador, que harmoniza e formata o conteúdo
    resposta = orquestrador_resp(user_id, session_id).invoke(
        {"input": resposta_str},
        config={"configurable": {"session_id": session}},
    )

    # Tenta extrair novamente o campo 'output', caso a resposta esteja em JSON
    if isinstance(resposta, str):
        try:
            resposta_json = json.loads(resposta)
            resposta = resposta_json.get("output", resposta_json)
        except json.JSONDecodeError:
            pass

    return resposta
