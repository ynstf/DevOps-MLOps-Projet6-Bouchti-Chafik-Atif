import requests
import time
import random

url = "http://localhost:8000/predict"

print("--- SCÉNARIO 3 : DATA DRIFT (Montants Anormaux) ---")
print("Simulation d'un changement radical de distribution des données...")
print("Regarde ton graphique 'Amount Distribution' ou 'Average Amount' !")
print("Appuie sur Ctrl+C pour arrêter.\n")

# Features V normales (Le comportement est clean)
normal_features = {f"V{i}": 0.1 for i in range(1, 29)}

while True:
    # DRIFT : Soudainement, les montants deviennent ÉNORMES
    # (Ex: Bug sur le site web ou blanchiment d'argent)
    amount = random.uniform(1000.0, 100000.0)

    data = normal_features.copy()
    data["Amount"] = amount

    try:
        response = requests.post(url, json=data)
        res_json = response.json()

        # Note : Le modèle peut marquer ça comme fraude à cause du montant (outlier)
        # C'est normal, mais le but ici est de voir la moyenne changer dans Grafana
        print(f"💰 DRIFT | Montant: {amount:.2f}€ | Fraude: {res_json['is_fraud']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    time.sleep(0.5)