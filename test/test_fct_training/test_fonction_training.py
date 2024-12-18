import unittest
import os
import shutil
import sys

# Ajoute le chemin absolu vers le dossier contenant fct.importation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fonctions.fct_model_training import *

# Fonctions à tester
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def data_split(df: pd.DataFrame):
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']
    numeric_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
    categorical_features = ['TYPE', 'BOROUGH']
    
    return X, y, numeric_features, categorical_features

def preprocess_data(features: pd.DataFrame, target: pd.Series, numeric_features: list, categorical_features: list):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor

def modelisation(features: pd.DataFrame, target: pd.Series, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Neural Network": MLPRegressor(random_state=42, max_iter=1000),
        "Linear Regression": LinearRegression()
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_rmse = rmse(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        results[name] = {
            "RMSE (test)": test_rmse,
            "R2 (test)": test_r2
        }

    for model, metrics in results.items():
        print(f"Modèle : {model}")
        print(f"  RMSE (test) : {metrics['RMSE (test)']:.2f}")
        print(f"  R2 (test) : {metrics['R2 (test)']:.2f}")
        print("-" * 30)

class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Préparer les fichiers de test pour les tests."""
        cls.valid_file = "test_valid_file.csv"
        cls.invalid_file = "test_invalid_file.csv"

        # Créer un fichier valide avec les colonnes TYPE, PRICE, BEDS, BATH, PROPERTYSQFT, BOROUGH
        valid_data = {
            "TYPE": ["Condo", "Villa", "House", "Penthouse"],
            "PRICE": [50000, 300000, 2000000, 10000000],
            "BEDS": [1, 3, 4, 5],
            "BATH": [1, 2, 3, 4],
            "PROPERTYSQFT": [500, 1500, 3000, 5000],
            "BOROUGH": ["Manhattan", "Queens", "Brooklyn", "Staten Island"]
        }
        pd.DataFrame(valid_data).to_csv(cls.valid_file, index=False)

        # Créer un fichier invalide (format incorrect, colonnes manquantes ou mal formatées)
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

    def test_valid_file(self):
        """Test sur un fichier valide."""
        print("\n[TEST] Test du fichier valide...")
        df = pd.read_csv(self.valid_file)
        X, y, numeric_features, categorical_features = data_split(df)
        preprocessor = preprocess_data(X, y, numeric_features, categorical_features)
        modelisation(X, y, preprocessor)

    def test_invalid_file(self):
        """Test sur un fichier invalide."""
        print("\n[TEST] Test du fichier invalide...")
        df = pd.read_csv(self.invalid_file)
        X, y, numeric_features, categorical_features = data_split(df)
        preprocessor = preprocess_data(X, y, numeric_features, categorical_features)
        modelisation(X, y, preprocessor)


if __name__ == "__main__":
    unittest.main()
