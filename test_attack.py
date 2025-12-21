import requests
import time
import random

url = "http://localhost:8000/predict"

print("--- SCÉNARIO 2 : ATTAQUE DE FRAUDE ---")
print("⚠️ Envoi massif de transactions frauduleuses...")
print("Regarde ton graphique 'Fraud Detection' grimper !")
print("Appuie sur Ctrl+C pour arrêter.\n")

# Template d'une FRAUDE (V14, V12, V10 très négatifs)
fraud_features = {f"V{i}": 0.0 for i in range(1, 29)}
fraud_features["V14"] = -8.5
fraud_features["V12"] = -7.0
fraud_features["V10"] = -5.0
fraud_features["V4"] = 4.0

while True:
    # Les fraudeurs essaient souvent des petits montants ou des montants nuls
    amount = random.uniform(0.0, 15.0)

    data = fraud_features.copy()
    data["Amount"] = amount

    # Variation légère pour ne pas envoyer exactement la même requête
    data["V14"] += random.uniform(-1.0, 1.0)

    try:
        response = requests.post(url, json=data)
        res_json = response.json()

        # On affiche en ROUGE si détecté
        status = "🔴 DÉTECTÉ" if res_json['is_fraud'] else "🟢 RATÉ"
        print(f"{status} | Montant: {amount:.2f}€ | Proba: {res_json['probability']:.4f}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    time.sleep(0.2)  # Vitesse rapide (Attaque)