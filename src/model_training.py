import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ajouter dynamiquement le chemin du dossier "fonctions" (revenir en arrière de deux niveaux)
# Add the parent directory of 'fonctions' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fonctions.fct_model_training import *

print("Chargement des données...")
data = pd.read_csv("data\data_clean\data_preprocess.csv")
print("Chargement des données terminé.\n")

print("Application de la fonction 'data_split'")
X, y, num_features, catg_features = data_split(df= data)

print("\nApplication de la fonction 'preprocess_data'")
transformer = preprocess_data(features = X, target = y, numeric_features = num_features, categorical_features = catg_features)

print("\nDébut processus de modélisation...")
modelisation(features = X, target = y, preprocessor = transformer)
print("Fin du processus de modélisation.")