import pandas as pd
import glob
import os
import os.path

def etl(path_folder_files : str) -> pd.DataFrame:
    
    full_data = pd.DataFrame()
    print("*"*50)
    print("Démarrage du processus d'importation\n")
    
    for fichier in glob.glob(f"{path_folder_files}\*.csv"):
        print(f"Importation du fichier {os.path.basename(fichier)}")
        df_temp = pd.read_csv(fichier, encoding='utf-8', compression='zip', sep = ',')
        
        full_data = pd.concat([full_data, df_temp]).reset_index(drop = True)
    
    print("Processus d'importation terminé")
    print("*"*50)
    print(f"\nAperçue des données :")
    print(full_data)
    return full_data