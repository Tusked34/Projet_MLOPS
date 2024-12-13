# MLOPS - Analyse des Prix de l'Immobilier à Londres

## Description du Projet

Ce projet universitaire vise à implémenter un pipeline MLOps pour analyser les prix de l'immobilier à Londres. Le modèle de Machine Learning (ML) développé permet de prédire les prix des propriétés en fonction de divers paramètres (emplacement, superficie, type de bien, etc.). Ce projet s'inscrit dans une démarche d'automatisation du processus de mise à jour, d'entraînement et de déploiement du modèle en production, tout en intégrant des pratiques MLOps pour garantir sa robustesse et sa scalabilité.

## Objectifs

- **Collecte des données** : Récupérer des données immobilières sur Londres provenant de différentes sources.
- **Prétraitement des données** : Nettoyer et préparer les données pour l'entraînement du modèle.
- **Entraînement du modèle ML** : Utiliser des modèles de régression (comme la régression linéaire, XGBoost, ou Random Forest) pour prédire les prix des propriétés.
- **Évaluation du modèle** : Mesurer la performance du modèle à l’aide de métriques adaptées telles que l'erreur quadratique moyenne (RMSE).
- **MLOps** : Automatiser le pipeline de ML, y compris le suivi des performances du modèle, l'intégration continue (CI) et la livraison continue (CD).

## Structure du Répertoire

```bash
MLOPS/
│
├── data/               # Contient les jeux de données bruts et prétraités
│   ├── raw/            # Données brutes
│   └── processed/      # Données après nettoyage et prétraitement
│
├── notebooks/          # Notebooks Jupyter pour l'exploration et le prototypage
│
├── src/                # Code source pour la préparation des données et l'entraînement des modèles
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── models/             # Modèles enregistrés et scripts d'évaluation
│   └── model.pkl
│
├── tests/              # Tests unitaires pour les scripts
│
├── requirements.txt    # Liste des dépendances Python
├── README.md           # Documentation du projet
└── Dockerfile          # Contient les instructions pour la création de l'image Docker
```
 
## Prérequis

- Python 3.7 ou supérieur
- Bibliothèques Python :
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`
  - `XGBoost`
  - `Flask`, etc.
- Docker (pour le déploiement du modèle en conteneur)

## Installation

1. Clonez ce repository :

   ```bash
   git clone https://github.com/votre-utilisateur/MLOPS.git
   cd MLOPS
   ```

2. Installez les dépendances :

``` bash
Copier le code
pip install -r requirements.txt
```

## Utilisation

1. Prétraiter les données :
```bash
Copier le code
python src/data_preprocessing.py
```

2. Entraîner le modèle :
```bash
Copier le code
python src/model_training.py
```
3. Évaluer les performances du modèle :
```bash
Copier le code
python src/model_evaluation.py
```

## Auteurs
Théo LAHMAR-CLAVIER
Djibrail Juhoor
Najim Djareddir