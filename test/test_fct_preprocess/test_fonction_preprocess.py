import unittest
import pandas as pd
import os
import shutil
import sys
# Ajoute le chemin absolu vers le dossier contenant fct.importation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fonctions.fct_preprocess import *

import os

class TestDataFrameProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialisation des fichiers de test."""
        cls.valid_file = "test_valid_file.csv"
        cls.invalid_file = "test_invalid_file.csv"

        # Création d'un fichier valide
        valid_data = {
            "BROKERTITLE": ["Broker1", "Broker2", "Broker3", "Broker4"],
            "TYPE": ["Condo", "Villa", "House", "Penthouse"],
            "PRICE": [50000, 300000, 2000000, 10000000],
            "BEDS": [1, 3, 4, 5],
            "BATH": [1, 2, 3, 4],
            "PROPERTYSQFT": [500, 1500, 3000, 5000],
            "ADDRESS": ["Addr1", "Addr2", "Addr3", "Addr4"],
            "STATE": ["NY", "NY", "NY", "NY"],
            "MAIN_ADDRESS": ["Main1", "Main2", "Main3", "Main4"],
            "ADMINISTRATIVE_AREA_LEVEL_2": ["New York County", "Queens County", "Kings County", "Richmond County"],
            "LOCALITY": ["Manhattan", "Queens", "Brooklyn", "Staten Island"],
            "SUBLOCALITY": ["Sub1", "Sub2", "Sub3", "Sub4"],
            "STREET_NAME": ["Street1", "Street2", "Street3", "Street4"],
            "LONG_NAME": ["Long1", "Long2", "Long3", "Long4"],
            "FORMATTED_ADDRESS": ["Fmt1", "Fmt2", "Fmt3", "Fmt4"],
            "LATITUDE": [40.7128, 40.7306, 40.6782, 40.5795],
            "LONGITUDE": [-74.0060, -73.9866, -73.9442, -74.1502]
        }
        pd.DataFrame(valid_data).to_csv(cls.valid_file, index=False)

        # Création d'un fichier invalide
        invalid_data = {
            "WRONG_COLUMN": ["Invalid", "Data", "Here"],
            "MISSING_PRICE": [None, None, None]
        }
        pd.DataFrame(invalid_data).to_csv(cls.invalid_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage des fichiers après les tests."""
        if os.path.exists(cls.valid_file):
            os.remove(cls.valid_file)
        if os.path.exists(cls.invalid_file):
            os.remove(cls.invalid_file)

    def test_valid_file_processing(self):
        """Test du pipeline de traitement sur un fichier valide."""
        df = pd.read_csv(self.valid_file)

        # Test assign_borough
        assign_borough(df)
        self.assertIn("BOROUGH", df.columns, "La colonne 'BOROUGH' doit être ajoutée.")
        self.assertTrue(
            all(df["BOROUGH"].isin(["Manhattan", "Brooklyn", "Queens", "Staten Island"])),
            "Les boroughs doivent être correctement assignés."
        )

        # Test drop_useless_columns
        drop_useless_columns(df)
        for col in ["BROKERTITLE", "STATE", "LATITUDE", "LONGITUDE", "ADDRESS", 
                    "LOCALITY", "SUBLOCALITY", "STREET_NAME", 
                    "ADMINISTRATIVE_AREA_LEVEL_2", "LONG_NAME", 
                    "FORMATTED_ADDRESS", "MAIN_ADDRESS"]:
            self.assertNotIn(col, df.columns, f"La colonne '{col}' doit être supprimée.")

        # Test filter_price_range
        filter_price_range(df)
        self.assertTrue(
            df["PRICE"].between(49000, 180000000).all(),
            "Les prix doivent être filtrés dans la plage valide."
        )

        # Test low_modalities_grouping
        low_modalities_grouping(df)
        self.assertIn("Autre_Type", df["TYPE"].unique(), "Les modalités rares doivent être regroupées.")

    def test_invalid_file_processing(self):
        """Test des erreurs sur un fichier invalide."""
        df = pd.read_csv(self.invalid_file)

        with self.assertRaises(KeyError):
            assign_borough(df)  # Colonne 'LOCALITY' absente.

        with self.assertRaises(KeyError):
            drop_useless_columns(df)  # Colonnes manquantes.

        with self.assertRaises(KeyError):
            filter_price_range(df)  # Colonne 'PRICE' absente.


if __name__ == "__main__":
    unittest.main()

