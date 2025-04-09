import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Charger le modèle entraîné (remplace 'model.pkl' par ton fichier si nécessaire)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialiser FastAPI
app = FastAPI()


# Définir le format des données d'entrée
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# Route pour tester l'API
@app.get("/")
def home():
    return {"message": "API de Détection de Fraude opérationnelle !"}


# Route pour prédire si une transaction est frauduleuse
@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(
        [[getattr(transaction, f"V{i}") for i in range(1, 29)] + [transaction.Amount]]
    )
    prediction = model.predict(data)[0]
    return {"fraudulent": bool(prediction)}
