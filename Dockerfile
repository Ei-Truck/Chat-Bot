# Usa imagem base leve e moderna do Python
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala dependências do sistema necessárias para compilar pacotes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip e instala as dependências Python
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Expõe a porta usada pela API
EXPOSE 5000

# Comando de inicialização
CMD ["python", "main.py"]
