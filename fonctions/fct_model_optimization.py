import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def optimisation_hyperparametres(features : pd.DataFrame, target : pd.Series, preprocessor, output_path_prediction : str):
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
        Rien.
        Ecrit un csv vers data\data_predict avec les valeurs réelles, prédites par le random forest
        et les valeurs prédites par le réseau de neurones.

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
        'model__n_estimators': [100, 500, 1000],
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
        ('model', MLPRegressor(random_state=42, max_iter=1000))
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
    
    # Prédictions avec les meilleurs modèles
    rf_best_pipeline = rf_grid.best_estimator_  # Pipeline avec les meilleurs paramètres pour Random Forest
    mlp_best_pipeline = mlp_grid.best_estimator_  # Pipeline avec les meilleurs paramètres pour MLP

    y_pred_rf = rf_best_pipeline.predict(X_test)
    y_pred_mlp = mlp_best_pipeline.predict(X_test)

    # Créer un DataFrame contenant les valeurs réelles et les prédictions
    predictions_df = pd.DataFrame({
        'Valeurs Réelles': y_test.values,
        'Prédictions RandomForest': y_pred_rf.round(2),
        'Prédictions RéseauNeurone': y_pred_mlp.round(2)
    })
    path_file = os.path.join(output_path_prediction, 'data_predicted.csv')
    predictions_df.to_csv(path_file, index=False)