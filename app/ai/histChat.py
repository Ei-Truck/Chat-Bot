import numpy as np

# Inicializando o modelo de embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
#Historico de chat no Redis  
class ChatHistory:
    def __init__(self):
        self.history = []
        self.embeddings = []
        
    def armazenar_mensagem(self,user,msg):
        self.history.append({"user":user,"mensage":msg})
        embedding = model.encode(msg)
        self.embeddings.append(embedding)
        
    def get_history(self):
        return self.history
    
    def search_history(self,query):
        query_embedding = model.encode(query)
        similarities = np.dot(self.embeddings,query_embedding)
        top_indicies = np.argsort(similarities)[::-1][:1]
        return [self.history[i] for i in top_indicies]