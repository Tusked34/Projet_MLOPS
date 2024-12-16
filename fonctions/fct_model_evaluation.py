from fonctions.fct_model_optimization import *
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualisation_predictions_from_df(predictions_df: pd.DataFrame, output_dir):
    """
    Visualise les prédictions des modèles à partir d'un DataFrame contenant les valeurs réelles et les prédictions.
    Génère des graphiques de parité, distribution des résidus et courbes des tendances pour chaque modèle.

    Args:
        predictions_df (pd.DataFrame): Un DataFrame contenant :
            - 'Valeurs Réelles' : Les valeurs réelles de la cible.
            - 'Prédictions RandomForest' : Les prédictions du modèle Random Forest.
            - 'Prédictions RéseauNeurone' : Les prédictions du modèle Réseau de Neurones.

    Returns:
        None: Affiche les graphiques.
    """
    
    # Vérification des colonnes requises
    required_columns = ['Valeurs Réelles', 'Prédictions RandomForest', 'Prédictions RéseauNeurone']
    for col in required_columns:
        if col not in predictions_df.columns:
            raise ValueError(f"Le DataFrame doit contenir la colonne '{col}'.")

    # Extraire les colonnes nécessaires
    y_true = predictions_df['Valeurs Réelles'].round(2)
    y_pred_rf = predictions_df['Prédictions RandomForest'].round(2)
    y_pred_mlp = predictions_df['Prédictions RéseauNeurone'].round(2)

    # Visualisation pour Random Forest
    print("Visualisation des prédictions pour Random Forest")
    _visualiser_predictions(y_true, y_pred_rf, title="Random Forest", output_dir=output_dir)

    # Visualisation pour Réseau de Neurones
    print("Visualisation des prédictions pour le Réseau de Neurones")
    _visualiser_predictions(y_true, y_pred_mlp, title="Réseau de Neurones", output_dir=output_dir)


def _visualiser_predictions(y_true, y_pred, title, output_dir):
    """
    Fonction utilitaire pour générer les graphiques de parité, résidus et tendances.

    Args:
        y_true (pd.Series): Valeurs réelles de la cible.
        y_pred (np.ndarray): Valeurs prédites par le modèle.
        title (str): Titre à afficher pour les graphiques.

    Returns:
        None: Affiche les graphiques.
    """
    
    # 1. Graphique de parité (Réel vs Prédit)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color="blue")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Ligne y=x
    plt.title(f"Graphique de Parité : Valeurs Réelles vs Prédites ({title})")
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Valeurs Prédites")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"parity_{title.replace(' ', '_')}.png"))
    plt.show()

    # 2. Histogramme des résidus (Erreurs)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.title(f"Distribution des Résidus ({title})")
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.axvline(0, color='red', linestyle='--')  # Ligne verticale à 0
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"residuals_{title.replace(' ', '_')}.png"))
    plt.show()

    # 3. Courbe des valeurs réelles et prédites
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.values, label="Valeurs Réelles", marker='o', linestyle='-', color="blue")
    plt.plot(y_pred, label="Valeurs Prédites", marker='x', linestyle='--', color="orange")
    plt.title(f"Comparaison des Tendances : Réel vs Prédit ({title})")
    plt.xlabel("Index")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"trend_{title.replace(' ', '_')}.png"))
    plt.show()