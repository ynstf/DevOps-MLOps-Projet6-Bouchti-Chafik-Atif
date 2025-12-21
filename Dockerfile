# 1. On part d'une image Python légère officielle
FROM python:3.9-slim

# 2. On définit le dossier de travail à l'intérieur du conteneur
WORKDIR /app

# 3. On copie d'abord les requirements pour profiter du cache Docker
# (Si tu changes ton code mais pas les libs, Docker ira plus vite)
COPY requirements.txt .

# 4. On installe les librairies
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout ton projet (app, model, etc.) dans le conteneur
COPY . .

# 6. On expose le port 8000 (celui de FastAPI)
EXPOSE 8000

# 7. La commande de démarrage
# On lance uvicorn sur l'host 0.0.0.0 (important pour Docker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]