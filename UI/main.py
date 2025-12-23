import uvicorn
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from sklearn.preprocessing import RobustScaler

# --- Métriques Prometheus ---
FRAUD_COUNTER = Counter(
    "fraud_detection_total",
    "Nombre total de fraudes détectées par le modèle"
)

AMOUNT_HISTOGRAM = Histogram(
    "transaction_amount_distribution",
    "Distribution des montants des transactions",
    buckets=[10, 50, 100, 500, 1000, 5000, 10000]
)

PROBABILITY_HISTOGRAM = Histogram(
    "fraud_probability_distribution",
    "Distribution des probabilités de fraude",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9]
)

# --- Configuration ---
# BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model_UnderS.pkl"
TEMPLATE_PATH = BASE_DIR / "templates"

model = None
scaler = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = RobustScaler()
        print(f"✅ Modèle chargé : {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur modèle : {e}")
else:
    print(f"⚠️ Modèle introuvable : {MODEL_PATH}")

app = FastAPI(title="Fraud Detection API", version="2.0.0")

# CORS pour permettre les requêtes depuis le navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring
Instrumentator().instrument(app).expose(app)

# --- Modèles ---
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
    """Preprocessing identique au notebook"""
    df = pd.DataFrame([transaction_dict])
    v_columns = [col for col in df.columns if col.startswith('V')]
    
    scaled_amount = scaler.fit_transform(df[['Amount']].values.reshape(-1, 1))
    scaled_time = scaler.fit_transform(df[['Time']].values.reshape(-1, 1))
    
    df['scaled_amount'] = scaled_amount.flatten()
    df['scaled_time'] = scaled_time.flatten()
    df = df.drop(['Time', 'Amount'], axis=1)
    
    final_columns = ['scaled_amount', 'scaled_time'] + v_columns
    df = df[final_columns]
    
    return df

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    html_path = TEMPLATE_PATH / "index.html"
    print(html_path)
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Interface not found</h1>"



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "UnderSampling LogisticRegression",
        "monitoring": "enabled"
    }

@app.post("/predict")
def predict(transaction: Transaction):
    """Prédiction pour une transaction unique"""
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        input_df = preprocess_transaction(transaction.model_dump())
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        is_fraud = bool(prediction)

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

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Prédiction pour un fichier CSV"""
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        results = []
        for _, row in df.iterrows():
            transaction_dict = row.to_dict()
            input_df = preprocess_transaction(transaction_dict)
            
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            is_fraud = bool(prediction)
            
            AMOUNT_HISTOGRAM.observe(row['Amount'])
            PROBABILITY_HISTOGRAM.observe(proba)
            
            if is_fraud:
                FRAUD_COUNTER.inc()
            
            results.append({
                "is_fraud": is_fraud,
                "probability": round(float(proba), 4),
                "risk_level": "CRITICAL" if proba > 0.8 else ("HIGH" if proba > 0.5 else "LOW")
            })
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
