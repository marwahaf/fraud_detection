# For permission issues with docker : newgrp docker
# Base image for Python 3.12
FROM python:3.12-slim

# Optimisation ? to check later 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app
COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip && \
pip install --no-cache-dir -r requirements.txt

# Exposer le port que FastAPI utilisera
EXPOSE 8000

# Lancer l'application FastAPI avec Uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# Command to run (teste à la fois l'entraînement et la prédiction)
# CMD ["sh", "-c", "python  src/predict.py"]
CMD ["python", "src/app.py"]