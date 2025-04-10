# fraud_detection

# 📊 Fraud Detection System  
Détection de fraudes sur transactions bancaires via ML (Random Forest/XGBoost).  

## 🚀 Fonctionnalités  
- Prétraitement automatique (SMOTE, StandardScaler).  
- Comparaison de 3 modèles (Logistic Regression, Random Forest, XGBoost).  
- API Flask intégrée (`app.py`).  

## 📦 Installation  
```bash
git clone https://github.com/marwahaf/fraud_detection
cd fraud_detection
pip install -r requirements.txt
python app/train.py  # Pour entraîner le modèle
