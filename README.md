
# 🏦 Projet MLOps : Détection de Fraude Bancaire en Temps Réel

## 📌 Introduction
Ce projet présente une architecture MLOps complète pour la détection de fraude bancaire en temps réel. Il couvre l'ensemble du cycle de vie d'un modèle de Machine Learning, depuis l'entraînement jusqu'au déploiement et au monitoring en production.

## 🎯 Objectifs
- **Modélisation robuste** : Entraîner un modèle capable de gérer un déséquilibre extrême des classes (fraudes < 0.2%)
- **Industrialisation** : Déployer le modèle via une API REST performante et conteneurisée
- **Observabilité** : Mettre en place un système de monitoring en temps réel avec alerting sur le Data Drift
- **Résilience** : Valider l'architecture via des tests de charge et scénarios d'attaque

## 🏗️ Architecture
Le projet suit une architecture MLOps modulaire en 4 phases :

```
Phase 1: Data Science & Modélisation
Phase 2: Industrialisation & Déploiement (Serving)
Phase 3: Observabilité & Monitoring
Phase 4: Validation Expérimentale
```

## 🛠️ Stack Technique
| Composant | Technologies |
|-----------|-------------|
| Langage | Python 3.9+ |
| ML Framework | Scikit-learn |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Conteneurisation | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Orchestration | Docker Compose |

## 📊 Dataset
- **Source** : Kaggle Credit Card Fraud Detection (ULB)
- **Volume** : 284,807 transactions
- **Fraudes** : 492 (0.172%)
- **Features** : 30 variables (V1-V28 via PCA, Time, Amount)

## 🔧 Installation et Déploiement

### Prérequis
- Docker & Docker Compose
- Python 3.9+ (pour le développement)

### Installation
```bash
# Cloner le repository
git clone <repo-url>
cd fraud-detection-mlops

# Construire et lancer les services
docker-compose up --build
```

### Services accessibles
- **API de prédiction** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
- **Grafana Dashboard** : http://localhost:3000
- **Prometheus** : http://localhost:9090

## 📈 Métriques de Performance

### Métriques Modèle
- **Recall (prioritaire)** : Maximiser la détection des vraies fraudes
- **F1-Score** : Équilibre entre précision et rappel
- **Latence API** : < 20ms (P99)

### Métriques Monitoring
- **Techniques** : Latence, débit, codes HTTP
- **Métier** : Taux de fraude, volume financier traité
- **Data Drift** : Distribution des montants, dérive statistique

## 🧪 Tests de Validation
Trois scénarios de validation ont été implémentés :

1. **Trafic standard** : Validation des performances nominales
2. **Attaque massive** : Test de résilience face aux fraudes simulées
3. **Data Drift** : Détection de dérive des distributions de données

## 📋 Fonctionnalités Clés
- ✅ API REST performante avec FastAPI
- ✅ Validation stricte des données avec Pydantic
- ✅ Conteneurisation complète avec Docker
- ✅ Monitoring temps réel avec Prometheus/Grafana
- ✅ Détection automatique du Data Drift
- ✅ Tests de charge et scénarios d'attaque
- ✅ Dashboard Grafana unifié

## 🚀 Perspectives d'Amélioration
- **Réentraînement automatique** : Pipeline de Continuous Training
- **CI/CD complet** : Intégration avec GitHub Actions
- **A/B Testing** : Déploiement progressif des nouveaux modèles
- **Persistance des données** : Base de données pour l'analyse des transactions

## 📚 Structure du Projet
```
fraud-detection-mlops/
├── api/                    # Code FastAPI
├── models/                 # Modèles entraînés
├── notebooks/              # Notebooks d'analyse
├── monitoring/             # Configuration Prometheus/Grafana
├── tests/                  # Tests et génération de trafic
├── docker-compose.yml      # Orchestration des services
└── requirements.txt        # Dépendances Python
```

## 👥 Auteurs
Projet développé par Bouchti, Chafik, Atif dans le cadre d'une étude MLOps avancée sur la détection de fraude bancaire.

## 📄 Licence
Ce projet est à des fins éducatives et démonstratives.

---

*Pour plus de détails techniques, consultez le rapport complet [Rapport_Detection_Des_Fraudes.docx].*
