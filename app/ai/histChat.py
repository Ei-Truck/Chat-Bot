import redis
import unicodedata
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def verifica_historico():
    resp=r.lrange("histChat",0,-1)
    return resp

def remover_caracteres_especiais(resposta: str) -> str:
    resposta = unicodedata.normalize("NFD", resposta)
    resposta = resposta.encode("ascii", "ignore").decode("utf-8")
    return resposta

def insere_resposta(resposta: str, respostasAnteriores: list, user_id: str):
    chave = get_hist_key(user_id)
    resp = json.loads(resposta)

    for campo in ['question', 'answer', 'judgmentAnswer']:
        if campo in resp:
            resp[campo] = remover_caracteres_especiais(resp[campo])

    if resp['status'] == 'Aprovado':
        answer = {
            "question": resp['question'],
            "answer": resp['answer']
        }
    elif resp['status'] == 'Reprovado':
        answer = {
            "question": resp['question'],
            "answer": resp['judgmentAnswer']
        }

    if not respostasAnteriores:
        r.rpush(chave, json.dumps(answer))
        r.expire(chave, TEMPO_EXPIRACAO)
        return 'Memória atualizada.'

    for x in respostasAnteriores:
        if json.loads(x) == answer:
            return 'Resposta já existente.'

    r.rpush(chave, json.dumps(answer))
    r.expire(chave, TEMPO_EXPIRACAO)
    return 'Memória atualizada.'

def deleta_historico(user_id: str):
    chave = get_hist_key(user_id)
    return r.delete(chave)