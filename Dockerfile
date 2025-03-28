# Base image for Python 3.12
FROM python:3.12-slim

# Set the working directory
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port que FastAPI utilisera
EXPOSE 8000

# Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]