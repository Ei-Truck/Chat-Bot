# Uso do MongoDB no Projeto

O *MongoDB* é utilizado para:

---

## 1. Armazenamento de histórico de conversas

Através da classe `MongoDBMessageHistory`, o MongoDB atua como **repositório persistente das interações do chatbot**.  
Cada documento contém:

- `SessionId`: identificador da sessão  
- `History`: JSON contendo as perguntas e respostas trocadas  
- `_id`: identificador gerado automaticamente (`ObjectId`)

Esse recurso garante que as conversas possam ser recuperadas posteriormente, preservando o contexto e o histórico de uso do sistema.

---

## 2. Armazenamento de embeddings (vetores numéricos)

O MongoDB também é utilizado **para armazenar os embeddings** gerados a partir de textos processados pelo modelo de linguagem.  
Cada embedding é representado por um vetor de números em ponto flutuante, que captura o significado semântico do texto original.

Esses embeddings são usados no processo de **RAG (Retrieval-Augmented Generation)**, permitindo:

- Busca por similaridade entre textos  
- Recuperação de informações contextuais  
- Geração de respostas mais precisas e contextualizadas  

Os documentos dessa collection contêm, por exemplo:

- `_id`: identificador único  
- `embedding`: vetor numérico (lista de floats, como mostrado na imagem)  
- `texto_original` ou `conteúdo_base`: texto de origem usado para gerar o embedding

---

### *Diagrama desenhado de utilização do MongoDB*:

![Diagrama MongoDB](./MongoDB.svg)
