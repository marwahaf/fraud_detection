# For permission issues with docker : newgrp docker
# Base image for Python 3.12
FROM python:3.12-slim

# # Optimisation ? to check later 
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV FLASK_APP=app.py


# Set the working directory
WORKDIR /app
COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip && \
pip install --no-cache-dir -r requirements.txt

# Exposer le port que FastAPI utilisera
EXPOSE 8000

CMD ["python", "src/app.py"]