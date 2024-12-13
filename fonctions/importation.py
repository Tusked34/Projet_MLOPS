import pandas as pd
import glob
import os

def etl(path_folder_files : str) -> pd.DataFrame:
    
    full_data = pd.DataFrame()
    print("*"*50)
    print("Démarrage du processus d'importation")
    
    for fichier in glob.glob(f"{path_folder_files}\*.csv"):
        fichier = open(fichier, "r", encoding = "utf-8")
        nom_fichier = os.path.basename(fichier).replace(".csv", "")
        print(f"Importation du fichier {nom_fichier}")
        
        df_temp = pd.read_csv(fichier, encoding='utf-8', compression='zip', sep = ',')
        
        full_data = pd.concat([full_data, df_temp])
    
    print("Processus d'importation terminé")
    print("*"*50)
    return full_data