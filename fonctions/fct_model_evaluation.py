from fonctions.fct_model_optimization import *
import matplotlib.pyplot as plt
import seaborn as sns

def visualisation_predictions(best_parameters_randomforest, best_parameters_reseau_neurone, preprocessor, X, y):
    """
    Visualise les prédictions des meilleurs modèles (Random Forest et Réseau de Neurones) 
    à l'aide de graphiques : parité, distribution des résidus, et courbe des tendances.

    Args:
        best_parameters_randomforest (dict): Meilleurs hyperparamètres pour Random Forest.
        best_parameters_reseau_neurone (dict): Meilleurs hyperparamètres pour le Réseau de Neurones.
        preprocessor (ColumnTransformer): Préprocesseur pour transformer les données.
        X (pd.DataFrame): Caractéristiques (features) du dataset.
        y (pd.Series): Variable cible (target).

    Returns:
        None: Affiche les graphiques des prédictions.
    """

    # Supprimer le préfixe "model__" des paramètres
    rf_params = {key.replace('model__', ''): value for key, value in best_parameters_randomforest.items()}
    mlp_params = {key.replace('model__', ''): value for key, value in best_parameters_reseau_neurone.items()}

    # Diviser les données en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Pipeline pour Random Forest avec les meilleurs hyperparamètres
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42, **rf_params))
    ])
    rf_pipeline.fit(X_train, y_train)  # Entraînement du modèle
    y_pred_rf = rf_pipeline.predict(X_test)  # Prédictions

    # 2. Pipeline pour Réseau de Neurones avec les meilleurs hyperparamètres
    mlp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(random_state=42, max_iter=500, **mlp_params))
    ])
    mlp_pipeline.fit(X_train, y_train)  # Entraînement du modèle
    y_pred_mlp = mlp_pipeline.predict(X_test)  # Prédictions

    # Visualisation pour Random Forest
    print("Visualisation des prédictions pour Random Forest")
    _visualiser_predictions(y_test, y_pred_rf, title="Random Forest")

    # Visualisation pour Réseau de Neurones
    print("Visualisation des prédictions pour le Réseau de Neurones")
    _visualiser_predictions(y_test, y_pred_mlp, title="Réseau de Neurones")

def _visualiser_predictions(y_test, y_pred, title):
    """
    Fonction utilitaire pour générer les graphiques de parité, résidus et tendances.

    Args:
        y_test (pd.Series): Valeurs réelles de la cible.
        y_pred (np.ndarray): Valeurs prédites par le modèle.
        title (str): Titre à afficher pour les graphiques.

    Returns:
        None: Affiche les graphiques.
    """

    # 1. Graphique de parité (Réel vs Prédit)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ligne y=x
    plt.title(f"Graphique de Parité : Valeurs Réelles vs Prédites ({title})")
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Valeurs Prédites")
    plt.grid(True)
    plt.show()

    # 2. Histogramme des résidus (Erreurs)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.title(f"Distribution des Résidus ({title})")
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.axvline(0, color='red', linestyle='--')  # Ligne verticale à 0
    plt.grid(True)
    plt.show()

    # 3. Courbe des valeurs réelles et prédites
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Valeurs Réelles", marker='o', linestyle='-', color="blue")
    plt.plot(y_pred, label="Valeurs Prédites", marker='x', linestyle='--', color="orange")
    plt.title(f"Comparaison des Tendances : Réel vs Prédit ({title})")
    plt.xlabel("Index")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.show()