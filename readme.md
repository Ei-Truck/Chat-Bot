
# 🚀 Documentação do Projeto Flask

Bem-vindo à documentação do projeto! Aqui você encontra tudo o que precisa pra rodar, entender e usar essa aplicação com IA no Flask.

---

## 📁 Estrutura do Projeto


```

app/
├── ai/
│   ├── text/
│   │   └── FAQ.txt
│   └── ai_model.py
│   └── embedding.py
│   └── histChat.py
├── config/
│   └── config.py
├── routes/
│   └── ai_route.py
├── schemas/
│   └── question_schema.py
├── service/
│   └── service\_ai.py
├── **init**.py

````

## ▶️ Como Rodar a Aplicação

### 📦 Pré-requisitos
- Python **3.8+**
- `pip` instalado

### 🛠️ Passo a Passo

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
    ````

2. Rode o servidor Flask:

   ```bash
   python main.py
   ```

3. Acesse via navegador:

   ```
   http://127.0.0.1:5000
   ```

---

## 🌐 Rotas da API

### ✅ **/health** — Verifica o status do servidor

* **Método:** `GET`
* **Requisição:** Nenhuma
* **Resposta:**

  ```json
  {
    "status": "OK"
  }
  ```
* **Status:** `200 OK`

---

### 💬 **/chat** — Envia uma pergunta para a IA

* **Método:** `POST`
* **Cabeçalhos:** `Content-Type: application/json`
* **Body:**

  ```json
  {
    "question": "Sua pergunta aqui",
    "user_id": 1
  }
  ```

#### ✔️ Resposta de Sucesso (`200`)

```json
{
  "timestamp": "2023-10-01T12:00:00",
  "content": {
    "answer": "Resposta gerada pela IA se aprovada, ou resposta gerada pelo juiz se desaprovada",
    "question": "Sua pergunta aqui"
  }
}
```

#### ❌ Erro de Validação (`400`)

```json
{
  "error": "field 'question' is required."
}
```

#### 🛑 Erro Interno (`500`)

```json
{
  "error": "Failed to generate answer"
}
```

---

## 🧩 Organização dos Pacotes

| Caminho        | Função                                                     |
| -------------- | ---------------------------------------------------------- |
| `app/ai/`      | Lógica da IA (modelo + arquivos auxiliares como `FAQ.txt`) |
| `app/routes/`  | Define as rotas da API com Flask                           |
| `app/schemas/` | Valida os dados de entrada usando `marshmallow`            |
| `app/service/` | Contém a lógica de negócio: fluxo de perguntas e respostas |

---

## 📝 Observações Finais

* ✅ Garanta que o `FAQ.txt` está preenchido corretamente.
* 🔒 A entrada é validada com rigor para evitar falhas.
* 🧠 Quer personalizar a IA? Mexa nos arquivos de `app/ai/`.

---

📌 **Dica:** Quer testar a API rapidinho? Use o [Postman](https://www.postman.com/)
