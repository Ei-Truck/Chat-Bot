import os
from dotenv import load_dotenv

# Carrega automaticamente as variáveis de ambiente do arquivo .env
load_dotenv()


# Classe de configuração da aplicação
class Config:
    # Define a chave da API Gemini a partir da variável de ambiente
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
