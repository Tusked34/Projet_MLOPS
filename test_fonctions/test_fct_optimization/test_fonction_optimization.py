import unittest
import os
import sys
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Ajoute le chemin absolu vers le dossier contenant fct.importation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fonctions.fct_model_optimization import *  # Assurez-vous d'importer correctement la fonction que vous testez

# Définition de la fonction d'optimisation des hyperparamètres
def optimisation_hyperparametres(features : pd.DataFrame, target : pd.Series, preprocessor, output_path_prediction : str):
    print("*" * 72)
    print("Optimisation des hyperparamètres du RandomForest et du Réseau de Neurone")
    print("*" * 72)
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

    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print("Optimisation du RandomForest terminée.\n")

    print("Optimisation du Réseau de neurone en cours ...")
    # Grille de recherche pour Réseau de Neurones
    mlp_params = {
        'model__hidden_layer_sizes': [(8,), (16, 16), (32,)],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate_init': [0.001, 0.01]
    }

    mlp_pipeline = Pipeline(steps=[ 
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(random_state=42, max_iter=1000))
    ])

    mlp_grid = GridSearchCV(mlp_pipeline, mlp_params, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
    mlp_grid.fit(X_train, y_train)
    print("Optimisation du Réseau de neurone terminée.")
    print("*" * 72)
    
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


# Classe de test unitaire
class TestOptimisationHyperparametres(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Préparer les fichiers de test pour les tests."""
        cls.valid_file = "test_valid_file.csv"
        cls.invalid_file = "test_invalid_file.csv"
        cls.output_dir = "data_predict"
        os.makedirs(cls.output_dir, exist_ok=True)

        # Créer un fichier valide avec les colonnes TYPE, PRICE, BEDS, BATH, PROPERTYSQFT, BOROUGH
        valid_data = {
            "TYPE": ["Condo", "Villa", "House", "Penthouse", "Condo", "Villa", "House", "Penthouse", "Condo", "Villa"],
            "PRICE": [50000, 300000, 2000000, 10000000, 60000, 350000, 2200000, 9500000, 70000, 380000],
            "BEDS": [1, 3, 4, 5, 2, 3, 4, 6, 1, 3],
            "BATH": [1, 2, 3, 4, 1, 2, 3, 5, 1, 2],
            "PROPERTYSQFT": [500, 1500, 3000, 5000, 600, 1600, 3100, 5100, 650, 1700],
            "BOROUGH": ["Manhattan", "Queens", "Brooklyn", "Staten Island", "Manhattan", "Queens", "Brooklyn", "Staten Island", "Manhattan", "Queens"]
        }
        pd.DataFrame(valid_data).to_csv(cls.valid_file, index=False)

        # Créer un fichier invalide (format incorrect, colonnes mal nommées)
        invalid_data = {
            "WRONG_COLUMN": ["Invalid", "Data", "Here"],
            "MISSING_PRICE": [None, None, None]
        }
        pd.DataFrame(invalid_data).to_csv(cls.invalid_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Nettoyer les fichiers après les tests."""
        if os.path.exists(cls.valid_file):
            os.remove(cls.valid_file)
        if os.path.exists(cls.invalid_file):
            os.remove(cls.invalid_file)
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_valid_file(self):
        """Test sur un fichier valide."""
        print("\n[TEST] Test du fichier valide...")
        df = pd.read_csv(self.valid_file)
        
        # Préparer X, y et le preprocessor sans utiliser data_split
        X = df.drop('PRICE', axis=1)
        y = df['PRICE']
        
        numeric_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
        categorical_features = ['TYPE', 'BOROUGH']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Appel à la fonction d'optimisation avec ces données
        optimisation_hyperparametres(X, y, preprocessor, self.output_dir)

    def test_invalid_file(self):
        """Test sur un fichier invalide."""
        print("\n[TEST] Test du fichier invalide...")
        df = pd.read_csv(self.invalid_file)
        try:
            # Essayer de traiter le fichier invalide
            X = df.drop('PRICE', axis=1)  # Cela échouera si 'PRICE' n'existe pas
            y = df['PRICE']  # Cela échouera aussi

            numeric_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
            categorical_features = ['TYPE', 'BOROUGH']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )

            optimisation_hyperparametres(X, y, preprocessor, self.output_dir)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier invalide : {e}")


if __name__ == "__main__":
    unittest.main()
