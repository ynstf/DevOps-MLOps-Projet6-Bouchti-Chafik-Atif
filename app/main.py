import uvicorn
import pandas as pd
import joblib
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ==============================================================================
# 1. DÉFINITION DES 5 MÉTRIQUES (LES MOUCHARDS INTELLIGENTS)
# ==============================================================================

# MÉTRIQUE 1 : Ratio Fraude/Safe (Labelisé)
# Permet de faire des camemberts : 99% Safe vs 1% Fraude
PREDICTION_COUNTER = Counter(
    "fraud_prediction_total",
    "Nombre de prédictions par type",
    ["status"]  # Label: 'fraud' ou 'safe'
)

# MÉTRIQUE 2 : Business Value (Compteur cumulatif)
# "Combien d'argent a transité par notre API ?"
PROCESSED_AMOUNT_COUNTER = Counter(
    "processed_amount_total",
    "Somme totale des montants traités "
)

# MÉTRIQUE 3 : Drift des Données (Montant)
# "Est-ce que les montants sont normaux ou bizarres ?"
# sum(rate(transaction_amount_dist_bucket[5m])) by (le)
AMOUNT_HISTOGRAM = Histogram(
    "transaction_amount_dist",
    "Distribution des montants (Détection de Drift)",
    buckets=[10, 100, 500, 1000, 5000, 10000, 50000]
)


# ==============================================================================
# 2. CONFIGURATION ET CHARGEMENT
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "pipeline_model_test.pkl"

model_pipeline = None
if MODEL_PATH.exists():
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé : {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
else:
    print(f"⚠️ Modèle introuvable : {MODEL_PATH}")

app = FastAPI(title="Fraud Detection API", version="2.0.0 Pro Metrics")

# Active /metrics pour Prometheus
Instrumentator().instrument(app).expose(app)

# Colonnes attendues par le modèle
EXPECTED_COLUMNS = ['Amount'] + [f'V{i}' for i in range(1, 29)]


class Transaction(BaseModel):
    Amount: float
    # Définition compacte des V1..V28
    V1: float;
    V2: float;
    V3: float;
    V4: float;
    V5: float
    V6: float;
    V7: float;
    V8: float;
    V9: float;
    V10: float
    V11: float;
    V12: float;
    V13: float;
    V14: float;
    V15: float
    V16: float;
    V17: float;
    V18: float;
    V19: float;
    V20: float
    V21: float;
    V22: float;
    V23: float;
    V24: float;
    V25: float
    V26: float;
    V27: float;
    V28: float


@app.get("/")
def health_check():
    return {"status": "ok", "metrics": "active"}


@app.post("/predict")
def predict(transaction: Transaction):
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Service indisponible (Modèle manquant)")

    # Début du chronomètre (Pour la métrique 5)
    start_time = time.time()

    try:
        # 1. Préparation Data
        input_data = transaction.model_dump()
        input_df = pd.DataFrame([input_data])
        # Réorganisation des colonnes pour correspondre au training
        input_df = input_df[EXPECTED_COLUMNS]

        # 2. Prédiction
        prediction = model_pipeline.predict(input_df)[0]
        proba = model_pipeline.predict_proba(input_df)[0][1]
        is_fraud = bool(prediction)

        # 3. Fin du chronomètre
        process_time = time.time() - start_time

        # ==========================================================
        # 🚨 MISE À JOUR DES MÉTRIQUES (Le Coeur du Monitoring)
        # ==========================================================

        # Métrique 1 : Compteur Fraud vs Safe
        status_label = "fraud" if is_fraud else "safe"
        PREDICTION_COUNTER.labels(status=status_label).inc()

        # Métrique 2 : Argent traité (On ajoute le montant au compteur total)
        PROCESSED_AMOUNT_COUNTER.inc(transaction.Amount)

        # Métrique 3 : Distribution des montants (Pour voir le Drift)
        AMOUNT_HISTOGRAM.observe(transaction.Amount)


        # ==========================================================

        return {
            "is_fraud": is_fraud,
            "probability": round(float(proba), 4),
            "process_time": round(process_time, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)