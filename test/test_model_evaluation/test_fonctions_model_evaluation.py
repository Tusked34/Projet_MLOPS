import unittest
import os
import shutil
import sys
# Ajoute le chemin absolu vers le dossier contenant fct.importation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fonctions.fct_model_evaluation import *  # Assurez-vous d'importer correctement la fonction que vous testez

class TestVisualisationPredictions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Préparer les fichiers de test pour les tests."""
        cls.valid_file = "test_predictions_valid_file.csv"
        cls.invalid_file = "test_predictions_invalid_file.csv"
        cls.output_dir = "output_predictions"
        os.makedirs(cls.output_dir, exist_ok=True)

        # Créer un fichier valide avec les colonnes 'Valeurs Réelles', 'Prédictions RandomForest', 'Prédictions RéseauNeurone'
        valid_data = {
            "Valeurs Réelles": [100000, 200000, 150000, 180000, 170000, 220000, 250000, 230000, 210000, 190000],
            "Prédictions RandomForest": [102000, 198000, 153000, 179000, 168000, 221000, 248000, 229000, 215000, 188000],
            "Prédictions RéseauNeurone": [101500, 201500, 148500, 181000, 169000, 219500, 249500, 232000, 208000, 189500]
        }
        df_valid = pd.DataFrame(valid_data)
        df_valid.to_csv(cls.valid_file, index=False)

        # Créer un fichier invalide (colonne manquante)
        invalid_data = {
            "Valeurs Réelles": [100000, 200000, 150000],
            "Prédictions RandomForest": [102000, 198000, 153000]
            # La colonne "Prédictions RéseauNeurone" est absente ici
        }
        df_invalid = pd.DataFrame(invalid_data)
        df_invalid.to_csv(cls.invalid_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Nettoyer les fichiers après les tests."""
        if os.path.exists(cls.valid_file):
            os.remove(cls.valid_file)
        if os.path.exists(cls.invalid_file):
            os.remove(cls.invalid_file)
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_visualisation_predictions(self):
        """Test de la fonction de visualisation des prédictions avec fichier valide et invalide."""
        
        # Tester le fichier valide
        print("\n[TEST] Test sur le fichier valide...")
        predictions_df_valid = pd.read_csv(self.valid_file)
        print("[TEST] Fichier valide chargé.")
        try:
            visualisation_predictions_from_df(predictions_df_valid, self.output_dir)
            print("[TEST] Graphiques générés avec succès pour le fichier valide.")
        except Exception as e:
            print(f"[TEST] Erreur lors de la génération des graphiques pour le fichier valide : {e}")

        # Tester le fichier invalide
        print("\n[TEST] Test sur le fichier invalide...")
        predictions_df_invalid = pd.read_csv(self.invalid_file)
        print("[TEST] Fichier invalide chargé.")
        try:
            visualisation_predictions_from_df(predictions_df_invalid, self.output_dir)
        except ValueError as e:
            print(f"[TEST] Erreur attendue pour le fichier invalide : {e}")
        except Exception as e:
            print(f"[TEST] Erreur inattendue pour le fichier invalide : {e}")

if __name__ == "__main__":
    unittest.main()
