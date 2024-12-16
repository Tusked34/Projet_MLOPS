import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# Définir une fonction pour calculer RMSE
def rmse(y_true, y_pred):
    """Calcule la moyenne des écarts quadratiques entre la valeur prédite et la valeur réelle.

    Args:
        y_true (int): y prédit correctement
        y_pred (int): y réel

    Returns:
        moyenne des carrés des erreur : mesure fréquemment utilisée des différences entre 
        les valeurs prédites par un modèle ou estimateur et les valeurs observées.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def data_split(df : pd.DataFrame):
    """Sépare et tri les données.

    Args:
        df (pd.DataFrame): dataframe pandas d'entrée, les données à splitées.

    Returns:
        X : Les variables explicatives
        y : La variable cible.
        numeric_features : les variables numériques.
        categorial_features : les variables catégorielles.
    """
    
    # Définir les caractéristiques et la cible
    X = df.drop(columns=['PRICE'])  # Caractéristiques (features)
    y = df['PRICE']                 # Cible (target)

    # Séparer les colonnes numériques et catégoriques
    numeric_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
    categorical_features = ['TYPE', 'BOROUGH']
    
    print("Séparation des variables :")
    print(f"X -> les variables explicatives : {[x for x in X.columns]}")
    print(f"y -> la variable d'intérêt, ici le prix : {[a for a in y[0:5]]} ")
    print(f"Les variables numériques : {numeric_features}")
    print(f"Les variables catégorielles : {categorical_features}")
    
    return X, y, numeric_features, categorical_features

def preprocess_data(features : pd.DataFrame, target : pd.Series, numeric_features : list, categorical_features : list):
    
    print("Construction d'un preprocessor pour transformer les données :\n")
    # Construire un transformer pour traiter les colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),   # Standardisation des colonnes numériques
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encodage OneHot pour les colonnes catégoriques
        ]
    )
    
    print(preprocessor)
    
    return preprocessor

def modelisation(features : pd.DataFrame, target : pd.Series, preprocessor):
    
    """
    Fonction pour modéliser les données à l'aide de différents modèles de régression.
    Applique un pipeline avec préprocesseur, ajuste les modèles sur les données d'entraînement
    et évalue les performances sur les données de test.

    Paramètres :
    - features : pd.DataFrame
        Les caractéristiques (features) utilisées pour entraîner les modèles.
    - target : pd.Series
        La cible (target) que les modèles essaieront de prédire.
    - preprocessor : transformer
        Un objet de type ColumnTransformer pour prétraiter les données.

    Retourne :
    - Aucun retour, les résultats sont affichés dans la console.
    """
    print("*"*60)
    print("*"*60)
    # Séparer les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    print("Séparation des données en données d'entrainement et de test.")
    print("Dimension des datasets :")
    print(f"--> X_train : {X_train.shape}")
    print(f"--> y_train : {y_train.shape}")
    print(f"--> X_test  : {X_test.shape}")
    print(f"--> y_test  : {y_test.shape}\n")

    # Définir les modèles
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Neural Network": MLPRegressor(random_state=42, max_iter=500),
        "Linear Regression": LinearRegression()
    }

    # Initialiser un dictionnaire pour stocker les résultats
    results = {}

    for name, model in models.items():
        # Définir le pipeline avec le préprocesseur et le modèle
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Ajuster le modèle sur les données d'entraînement
        pipeline.fit(X_train, y_train)
        
        # Prédire sur les données de test
        y_pred = pipeline.predict(X_test)
        
        # Calculer les métriques sur le jeu de test
        test_rmse = rmse(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        # Stocker les résultats
        results[name] = {
            "RMSE (test)": test_rmse,
            "R2 (test)": test_r2
        }

    # Afficher les résultats
    print("Résultats des Modèles :")
    for model, metrics in results.items():
        print(f"Modèle : {model}")
        print(f"  RMSE (test) : {metrics['RMSE (test)']:.2f}")
        print(f"  R2 (test) : {metrics['R2 (test)']:.2f}")
        print("-" * 30)
    
    print("*"*60)
    print("*"*60)