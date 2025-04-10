import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


if not os.path.exists("models/model.pkl") or not os.path.exists("models/scaler.pkl"):
    print("Veuillez d'abord exécuter train.py")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    prediction = model.predict(df)[0]
    probabilite = model.predict_proba(df)[0][1]

    return {"fraud": bool(prediction), "probability": probabilite}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)  # Lance l'API sur le port 5000
