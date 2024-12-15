import pandas as pd
import numpy as np

import sys
import os

# Ajouter dynamiquement le chemin du dossier "fonctions" (revenir en arrière de deux niveaux)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fonctions.importation import *
from fonctions.fct_preprocess import *

### main :
df = etl(r"data\data_raw")

assign_borough(df)
drop_useless_columns(df)
low_modalities_grouping(df)
print("\nData cleaning OK")

df.to_csv(r'data\data_clean\data_preprocess.csv', index=False)
print("\nClean data saved OK")