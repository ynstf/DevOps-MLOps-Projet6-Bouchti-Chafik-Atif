import uvicorn
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from sklearn.preprocessing import RobustScaler

# --- 1. Définition des Métriques Prometheus ---
FRAUD_COUNTER = Counter(
    "fraud_detection_total",
    "Nombre total de fraudes détectées par le modèle"
)

AMOUNT_HISTOGRAM = Histogram(
    "transaction_amount_distribution",
    "Distribution des montants des transactions (Drift Detection)",
    buckets=[10, 50, 100, 500, 1000, 5000, 10000]
)

PROBABILITY_HISTOGRAM = Histogram(
    "fraud_probability_distribution",
    "Distribution des probabilités de fraude prédites",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9]
)

# --- 2. Configuration & Chargement Modèle ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model_UnderS.pkl"

model = None
scaler = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        # Initialize scaler (will be fitted on first batch or loaded if saved)
        scaler = RobustScaler()
        print(f"✅ Modèle chargé : {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur modèle : {e}")
else:
    print(f"⚠️ Modèle introuvable : {MODEL_PATH}")

app = FastAPI(title="Fraud Detection API", version="2.0.0 (UnderSampling Model)")

# --- 3. Activation du Monitoring ---
Instrumentator().instrument(app).expose(app)

# --- 4. Schéma de Données ---
class Transaction(BaseModel):
    Time: float
    Amount: float
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


def preprocess_transaction(transaction_dict: dict) -> pd.DataFrame:
    """
    Applique le même preprocessing que dans le notebook:
    1. Scale Amount et Time avec RobustScaler
    2. Renomme en scaled_amount et scaled_time
    3. Drop les colonnes originales
    4. Réorganise les colonnes (scaled en premier)
    """
    # Créer le DataFrame
    df = pd.DataFrame([transaction_dict])
    
    # Sauvegarder les colonnes V1-V28
    v_columns = [col for col in df.columns if col.startswith('V')]
    
    # Scale Amount et Time
    scaled_amount = scaler.fit_transform(df[['Amount']].values.reshape(-1, 1))
    scaled_time = scaler.fit_transform(df[['Time']].values.reshape(-1, 1))
    
    # Ajouter les colonnes scaled
    df['scaled_amount'] = scaled_amount.flatten()
    df['scaled_time'] = scaled_time.flatten()
    
    # Supprimer Time et Amount
    df = df.drop(['Time', 'Amount'], axis=1)
    
    # Réorganiser: scaled_amount, scaled_time, puis V1-V28
    final_columns = ['scaled_amount', 'scaled_time'] + v_columns
    df = df[final_columns]
    
    return df


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "UnderSampling LogisticRegression",
        "monitoring": "enabled"
    }


@app.post("/predict")
def predict(transaction: Transaction):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Preprocessing identique au notebook
        input_df = preprocess_transaction(transaction.model_dump())

        # Prédiction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        is_fraud = bool(prediction)

        # --- MISE À JOUR DES MÉTRIQUES ---
        AMOUNT_HISTOGRAM.observe(transaction.Amount)
        PROBABILITY_HISTOGRAM.observe(proba)
        
        if is_fraud:
            FRAUD_COUNTER.inc()

        return {
            "is_fraud": is_fraud,
            "probability": round(float(proba), 4),
            "risk_level": "CRITICAL" if proba > 0.8 else ("HIGH" if proba > 0.5 else "LOW"),
            "model_version": "UnderSampling"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info")
def model_info():
    """Endpoint pour vérifier les informations du modèle"""
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": str(type(model).__name__),
        "features_expected": 30,  # scaled_amount, scaled_time + V1-V28
        "preprocessing": "RobustScaler on Amount & Time"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)