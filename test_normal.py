import requests
import time
import random

url = "http://localhost:8000/predict"

print("--- SCÉNARIO 1 : TRAFIC NORMAL ---")
print("Simulation de clients standards qui font leurs courses...")
print("Appuie sur Ctrl+C pour arrêter.\n")

# Template d'une transaction normale (V proches de 0)
base_data = {f"V{i}": 0.1 for i in range(1, 29)}

while True:
    # Montant réaliste : entre 10€ et 100€
    amount = random.uniform(10.0, 100.0)

    # On ajoute un peu de bruit aléatoire aux V-features
    data = base_data.copy()
    data["Amount"] = amount
    for k in data:
        if k != "Amount":
            data[k] += random.uniform(-0.5, 0.5)

    try:
        response = requests.post(url, json=data)
        res_json = response.json()
        print(f"✅ Montant: {amount:.2f}€ | Fraude: {res_json['is_fraud']} | Proba: {res_json['probability']:.4f}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    time.sleep(0.5)  # Vitesse modérée