import unittest
import pandas as pd
import os
import shutil
import sys

# Ajoute le chemin absolu vers le dossier contenant fct.importation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Vérifiez que le chemin est bien ajouté
print(sys.path)

from fonctions.fct_importation import *

class TestETLFunction(unittest.TestCase):
    def setUp(self):
        """
        Prépare un environnement de test avec des fichiers CSV.
        """
        self.test_folder = "test_data"
        os.makedirs(self.test_folder, exist_ok=True)

        data1 = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n"
        data2 = "col1,col2,col3\n16,17,18\n19,20,21\n22,23,24\n25,26,27\n28,29,30\n31,32,33\n34,35,36\n37,38,39\n"

        with open(os.path.join(self.test_folder, "file1.csv"), "w") as f:
            f.write(data1)

        with open(os.path.join(self.test_folder, "file2.csv"), "w") as f:
            f.write(data2)

    def tearDown(self):
        """
        Nettoie le dossier de test après les tests.
        """
        shutil.rmtree(self.test_folder)

    def test_etl_multiple_files(self):
        """
        Teste si la fonction charge correctement plusieurs fichiers CSV.
        """
        df = etl(self.test_folder)
        self.assertIsInstance(df, pd.DataFrame, "Le résultat doit être un DataFrame.")
        self.assertEqual(df.shape, (13, 3), "Le DataFrame doit contenir 13 lignes et 3 colonnes.")

    def test_etl_empty_directory(self):
        """
        Teste si la fonction gère correctement un dossier vide.
        """
        os.makedirs("empty_folder", exist_ok=True)
        df = etl("empty_folder")
        self.assertTrue(df.empty, "Le DataFrame doit être vide pour un dossier vide.")
        shutil.rmtree("empty_folder")

    def test_etl_invalid_path(self):
        """
        Teste si la fonction lève une exception pour un chemin invalide.
        """
        with self.assertRaises(Exception):
            etl("chemin/inexistant")

    def test_etl_invalid_file_content(self):
        """
        Teste si la fonction gère correctement un fichier mal formaté.
        """
        invalid_file = os.path.join(self.test_folder, "invalid.csv")
        with open(invalid_file, "w") as f:
            f.write("col1,col2\n1,2,3\n4,5")  # Fichier mal formaté

        with self.assertRaises(Exception):
            etl(self.test_folder)

if __name__ == "__main__":
    unittest.main()
