# Imagem base Python
FROM python:3.9-slim

# Instala o Nginx
RUN apt-get update && apt-get install -y nginx curl && rm -rf /var/lib/apt/lists/*

# Cria diretórios para a API e Nginx
WORKDIR /app
COPY . /app

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Configura Nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expõe a porta 80
EXPOSE 80

# Script para rodar API + Nginx
CMD ["bash", "-c", "python main.py & nginx -g 'daemon off;'"]
