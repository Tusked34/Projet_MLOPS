import pandas as pd
import glob
import os
import os.path

def import_packages():
    # Importation des données + Pre processing
    import pandas as pd
    import glob
    import os
    import sys
    import os.path
    import warnings
    warnings.filterwarnings("ignore")
    
    # Stat Desc
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Entrainement + Evaluation du Modèle
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import make_scorer, mean_squared_error, r2_score




def etl(path_folder_files : str) -> pd.DataFrame:
    """
    Effectue un processus ETL (Extract, Transform, Load) en important tous les fichiers CSV compressés
    situés dans un dossier spécifié, puis les concatène dans un seul DataFrame.

    Cette fonction lit tous les fichiers CSV compressés en ZIP dans le dossier spécifié, 
    les charge dans des DataFrames temporaires, et les combine en un DataFrame unique.

    Args:
        path_folder_files (str): Chemin vers le dossier contenant les fichiers CSV compressés à importer.
    
    Returns:
        pd.DataFrame: Un DataFrame contenant la concaténation des données de tous les fichiers importés.
    
    Prints:
        Messages d'étape du processus d'importation, ainsi qu'un aperçu des données résultantes.
    """
    full_data = pd.DataFrame()
    print("*"*50)
    print("Démarrage du processus d'importation\n")
    
    for fichier in glob.glob(f"{path_folder_files}\*.csv"):
        print(f"Importation du fichier {os.path.basename(fichier)}")
        df_temp = pd.read_csv(fichier, encoding='utf-8', sep = ',')
        
        full_data = pd.concat([full_data, df_temp]).reset_index(drop = True)
    
    print("Processus d'importation terminé")
    print("*"*50)
    print(f"Dimension du dataset : {full_data.shape[0]} lignes et {full_data.shape[1]} colonnes")
    return full_data




