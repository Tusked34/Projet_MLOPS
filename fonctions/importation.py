import pandas as pd
import glob
import os
import os.path

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
        df_temp = pd.read_csv(fichier, encoding='utf-8', compression='zip', sep = ',')
        
        full_data = pd.concat([full_data, df_temp]).reset_index(drop = True)
    
    print("Processus d'importation terminé")
    print("*"*50)
    print(f"Dimension du dataset : {full_data.shape[0]} lignes et {full_data.shape[1]} colonnes")
    return full_data