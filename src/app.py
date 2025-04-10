import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../static')
CORS(app)  # Active CORS pour toutes les routes

# Charge le modèle et le scaler au démarrage
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return send_from_directory('../static', "index.html")

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # Pré-vol CORS
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        return response

    data = request.json
    
    # Crée un dictionnaire avec toutes les features attendues
    features = {
        'V1': data.get('V1', 0),
        'V2': data.get('V2', 0),
        'V3': 0, 'V4': 0, 'V5': 0, 'V6': 0, 'V7': 0, 'V8': 0, 'V9': 0,
        'V10': 0, 'V11': 0, 'V12': 0, 'V13': 0, 'V14': 0, 'V15': 0,
        'V16': 0, 'V17': 0, 'V18': 0, 'V19': 0, 'V20': 0, 'V21': 0,
        'V22': 0, 'V23': 0, 'V24': 0, 'V25': 0, 'V26': 0, 'V27': 0,
        'V28': 0,
        'Amount': data['Amount']
    }
    
    df = pd.DataFrame([features])
    df["Amount"] = scaler.transform(df[["Amount"]])  # Normalisation
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    response = jsonify({
        "fraud": bool(prediction),
        "probability": float(probability)
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)