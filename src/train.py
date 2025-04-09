import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
import os
import pickle


# ---  Charger le dataset ---
DATA_PATH = os.path.join("data", "creditcard.csv")
if not os.path.exists(DATA_PATH):
    df = pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    )
    df.to_csv(DATA_PATH, index=False)
    print("Dataset téléchargé dans 'data/'")
else:
    df = pd.read_csv(DATA_PATH)
    print("Dataset présent dans 'data/'")

# --- Analyse exploratoire ---
df.head()  # Afficher les premières lignes
df.describe()  # Statistiques descriptives

# Nombre de fraudes vs transactions normales
fraud_count = df["Class"].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_count.index, y=fraud_count.values, palette="coolwarm")
plt.title("Répartition des transactions frauduleuses vs normales")
plt.xticks([0, 1], ["Normal (0)", "Fraude (1)"])
plt.show()

# Correlation des variables
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# --- Pré-traitement ---

# Vérifier s'il y a des valeurs manquantes
missing_values = df.isnull().sum().sum()
print(f"Nombre total de valeurs manquantes : {missing_values}")

# Vérifier s'il y a des valeurs aberrantes
outliers = df.describe().loc["25%"] - df.describe().loc["75%"]
print(f"Nombre total de valeurs aberrantes : {outliers.sum()}")

# Standardiser la colonne 'Amount'
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(columns=["Time"])  # colonne 'Time' n'apporte pas de valeur utile

# Sauvegarde du scaler pour la prédiction
with open(os.path.join("models", "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)


# --- Rééquilibrage des Classes ---
def under_sampling(df):
    """Under-sampling : Réduire le nombre de transactions normales."""
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
    """Sur-échantillonnage (SMOTE) : Générer plus de fraudes artificielles."""
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

#  --- Entrainement du model  ---

# Séparer les features (X) et la cible (y)
X = df_balanced.drop(columns=["Class"])  # Variables explicatives
y = df_balanced["Class"]  # Variable cible (0 = normal, 1 = fraude)
# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Taille du train : {X_train.shape}, Taille du test : {X_test.shape}")

# 3 Algorithmes :
# Régression Logistique (baseline)
# Random Forest(bon compromis entre simplicité et performance)
# XGBoost(souvent performant sur ce genre de problème)

# M1 : Entraînement du modèle
def Logisitic_Regression(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Prédiction
    y_pred_log = log_reg.predict(X_test)

    # Évaluation
    print("🔹 Régression Logistique 🔹")
    print(classification_report(y_test, y_pred_log))
    return log_reg

# M2 : Entraînement du modèle
def Random_Forest(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # Prédiction
    y_pred_rfc = rfc.predict(X_test)

    # Évaluation
    print("🔹 Random Forest 🔹")
    print(classification_report(y_test, y_pred_rfc))
    return rfc

# M3 : Entraînement du modèle
def XGBoost(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    # Prédiction
    y_pred_xgb = xgb.predict(X_test)

    # Évaluation
    print("🔹 XGBoost 🔹")
    print(classification_report(y_test, y_pred_xgb))
    return xgb 

# Established that random forest is better here.
log_reg =Logisitic_Regression(X_train, X_test, y_train, y_test)
rfc = Random_Forest(X_train, X_test, y_train, y_test)
xgb = XGBoost(X_train, X_test, y_train, y_test)

# Sauvegarde du modèle
with open("models/model.pkl", "wb") as file:
    pickle.dump(rfc, file)  # Remplace "xgb" par ton meilleur modèle
