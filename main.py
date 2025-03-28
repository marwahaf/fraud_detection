import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


# Charger le dataset
df = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)

# Afficher les premières lignes
df.head()

# Statistiques descriptives
df.describe()

# Nombre de fraudes vs transactions normales
fraud_count = df["Class"].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_count.index, y=fraud_count.values, palette="coolwarm")
plt.title("Répartition des transactions frauduleuses vs normales")
plt.xticks([0, 1], ["Normal (0)", "Fraude (1)"])
plt.show()

# Correlation des variables
# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.show()

# Vérifier s'il y a des valeurs manquantes
missing_values = df.isnull().sum().sum()
print(f"Nombre total de valeurs manquantes : {missing_values}")

# Vérifier s'il y a des valeurs aberrantes
outliers = df.describe().loc["25%"] - df.describe().loc["75%"]
print(f"Nombre total de valeurs aberrantes : {outliers.sum()}")

# Standardiser la colonne 'Amount'
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Supprimer la colonne 'Time' qui n'apporte pas de valeur utile
df = df.drop(columns=["Time"])

# Under-sampling : Réduire le nombre de transactions normales.
# Sur-échantillonnage (SMOTE) : Générer plus de fraudes artificielles.
def under_sampling(df):
    # Séparer les classes
    df_fraud = df[df["Class"] == 1]  # Fraudes
    df_normal = df[df["Class"] == 0]  # Transactions normales

    # Ré-échantillonner pour avoir autant de normales que de fraudes
    df_normal_under = resample(
        df_normal, replace=False, n_samples=len(df_fraud), random_state=42
    )

    # Reconstituer un dataset équilibré
    df_balanced = pd.concat([df_normal_under, df_fraud])

    # Vérifier la répartition
    print(df_balanced["Class"].value_counts())
    return df_balanced

def over_sampling(df):
    # Séparer les features et la target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Appliquer SMOTE
    smote = SMOTE(
        sampling_strategy=0.5, random_state=42
    )  # On augmente les fraudes à 50% des normales
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Vérifier la nouvelle répartition
    print(pd.Series(y_resampled).value_counts())
    return pd.concat([X_resampled, y_resampled], axis=1)

df_balanced = under_sampling(df)
df_resampled = over_sampling(df)

# Séparer les features (X) et la cible (y)
X = df_balanced.drop(columns=["Class"])  # Variables explicatives
y = df_balanced["Class"]  # Variable cible (0 = normal, 1 = fraude)

# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Taille du train : {X_train.shape}, Taille du test : {X_test.shape}")

# Entrainement du model
# 3 Algorithmes :  
# Régression Logistique (baseline)
# Random Forest(bon compromis entre simplicité et performance) 
# XGBoost(souvent performant sur ce genre de problème)

# M1 : Entraînement du modèle
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prédiction
y_pred_log = log_reg.predict(X_test)

# Évaluation
print("🔹 Régression Logistique 🔹")
print(classification_report(y_test, y_pred_log))

# M2 : Entraînement du modèle
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)           

# Prédiction
y_pred_rfc = rfc.predict(X_test)

# Évaluation
print("🔹 Random Forest 🔹")
print(classification_report(y_test, y_pred_rfc))

# M3 : Entraînement du modèle
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Prédiction
y_pred_xgb = xgb.predict(X_test)

# Évaluation    
print("🔹 XGBoost 🔹")
print(classification_report(y_test, y_pred_xgb))

import pickle

# Sauvegarde du modèle
with open("model.pkl", "wb") as file:
    pickle.dump(rfc, file)  # Remplace "xgb" par ton meilleur modèle

