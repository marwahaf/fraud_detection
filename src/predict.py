import os
import pickle

import pandas as pd

if not os.path.exists("models/model.pkl") or not os.path.exists("models/scaler.pkl"):
    print("Veuillez d'abord exécuter train.py")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Exemple test
new_data = {
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62,  # Montant original (sera normalisé automatiquement)
}

new_df = pd.DataFrame([new_data])
new_df["Amount"] = scaler.fit_transform(new_df[["Amount"]])

# Predict
prediction = model.predict(new_df)
probabilite = model.predict_proba(new_df)

print(prediction)
print("resultat de le predicition")
print("Fraude") if prediction[0] == 1 else print("Normal")
