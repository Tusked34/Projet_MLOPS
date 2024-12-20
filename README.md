# MLOPS - Analyse des Prix de l'Immobilier à New-York

## Description du Projet

Ce projet universitaire vise à implémenter un pipeline MLOps pour analyser les prix de l'immobilier à New-York. Plusieurs modèles de Machine Learning développés permettent de prédire les prix des propriétés en fonction de divers paramètres (emplacement, superficie, type de bien, etc.). Ce projet s'inscrit dans une démarche d'automatisation du processus de mise à jour, d'entraînement et de déploiement du modèle en production en intégrant des pratiques MLOps pour garantir sa scalabilité.

## Objectifs

- **Collecte des données** : Récupérer des données immobilières sur New-York.
- **Prétraitement des données** : Nettoyer et préparer les données pour l'entraînement du modèle.
- **Entraînement du modèle ML** : Utiliser des modèles de régression pour prédire les prix des propriétés.
- **Évaluation du modèle** : Mesurer la performance du modèle à l’aide de métriques adaptées.
- **MLOps** : Automatiser le pipeline de ML, y compris le suivi des performances du modèle, l'intégration continue (CI) et la livraison continue (CD).

## Structure du Répertoire

```bash
MLOPS/
│
├── data/  # Jeux de données 
│   ├── data_raw/   # Données brutes
│   ├── data_predict/   # Données avec les prédictions du modèle
│   └── data_clean/ # Données après nettoyage et prétraitement
│
├── fonctions/   # Bibliothèques des fonctions utilisés
│   ├── Doc_Fonctions.html # Documentation des Fonctions
│   ├── fct_importation.py
│   ├── fct_stat_desc.py
│   ├── fct_preprocess.py
│   ├── fct_model_training.py
│   ├── fct_model_optimization.py
│   └── fct_model_evaluation.py
│
├── notebooks/
│   └── Demo.ipynb  # Notebook de démonstration du projet
│
├── results/ # Images issues des visualisations des résultats des modèles
│
├── src/  # Code source pour la préparation des données et l'entraînement des modèles
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_optimization.py
│   └── model_evaluation.py
│
├── tests/  # Tests unitaires des différentes fonctions
│
├── requirements.txt  # Liste des dépendances Python
└── README.md  # Documentation du projet
```


## Prérequis

- Python 3.0 ou supérieur
- Bibliothèques Python :
  - `Pandas`
  - `NumPy`
  - `MathPlotLib`
  - `Scikit-learn`
  -  etc.


## Installation

1. Clonez ce repository :

   ```bash
   git clone https://github.com/Tusked34/Projet_MLOPS.git
   ```

2. Installez les dépendances :

``` bash
pip install -r requirements.txt
```

## Utilisation

1. Prétraiter les données :
```bash
python src/data_preprocessing.py
```

2. Entraîner le modèle :
```bash
python src/model_training.py
```

3. Optimiser les parametres du modèle :
```bash
python src/model_optimization.py
```

4. Évaluer les performances du modèle :
```bash
python src/model_evaluation.py
```

## Auteurs
Théo LAHMAR-CLAVIER, 
Djibrail JUHOOR, 
Najim DJAREDDIR, 
Clémentine LEPEZ