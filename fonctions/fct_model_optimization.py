import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def optimisation_hyperparametres(features : pd.DataFrame, target : pd.Series, preprocessor):
    """
    Optimise les hyperparamètres de deux modèles : Random Forest et Réseau de Neurones,
    en utilisant une recherche sur grille avec validation croisée.

    Cette fonction applique un pipeline incluant un préprocesseur (prétraitement des données) 
    et optimise les hyperparamètres des deux modèles via `GridSearchCV`. Elle retourne les 
    meilleurs hyperparamètres pour chaque modèle et affiche les résultats de la recherche.

    Args:
        features (pd.DataFrame): Les caractéristiques (features) utilisées pour entraîner les modèles.
        target (pd.Series): La variable cible (target) que les modèles essaieront de prédire.
        preprocessor (ColumnTransformer): Un objet de type ColumnTransformer appliquant le prétraitement 
                                          des données (standardisation, encodage, etc.).

    Returns:
        tuple: Un tuple contenant :
            - dict : Les meilleurs hyperparamètres pour Random Forest (`meilleurs_estimators_rf`).
            - dict : Les meilleurs hyperparamètres pour le Réseau de Neurones (`meilleurs_estimators_mlp`).

    Prints:
        - Les étapes de l'optimisation.
        - Les meilleurs paramètres et scores obtenus pour chaque modèle.
    """
    
    print("*"*72)
    print("Optimisation des hyperparamètres du RandomForest et du Réseau de Neurone")
    print("*"*72)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    print("Optimisation du RandomForest en cours ...")
    # Grille de recherche pour Random Forest
    rf_params = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [10, 20, None]
    }

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv = 4, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print("Optimisation du RandomForest terminée.\n")

    print("Optimisation du Réseau de neurone en cours ...")
    # Grille de recherche pour Réseau de Neurones
    mlp_params = {
        'model__hidden_layer_sizes': [(8,), (16, 16), (32,)],
        'model__alpha': [0.0001, 0.001, 0.01],  # Regularization term
        'model__learning_rate_init': [0.001, 0.01]
    }

    mlp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(random_state=42, max_iter=500))
    ])

    mlp_grid = GridSearchCV(mlp_pipeline, mlp_params, cv = 4, scoring='neg_mean_squared_error', n_jobs=-1)
    mlp_grid.fit(X_train, y_train)
    print("Optimisation du Réseau de neurone terminée.")
    print("*"*72)
    
    # Meilleurs hyperparamètres et performances
    print("\nRésultats de l'optimisation des hyperparamètres :")
    print(f"Random Forest - Meilleurs paramètres : {rf_grid.best_params_}")
    print(f"Random Forest - Meilleur RMSE (train cross-val) : {-rf_grid.best_score_:.2f}")

    print(f"Neural Network - Meilleurs paramètres : {mlp_grid.best_params_}")
    print(f"Neural Network - Meilleur RMSE (train cross-val) : {-mlp_grid.best_score_:.2f}")
    
    meilleurs_estimators_rf = rf_grid.best_params_
    meilleurs_estimators_mlp = mlp_grid.best_params_
    
    return meilleurs_estimators_rf, meilleurs_estimators_mlp