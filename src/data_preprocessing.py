import pandas as pd
import numpy as np

import sys
import os

# Ajoute le chemin du dossier parent à sys.path
notebook_dir = os.path.abspath('..')  # Dossier parent de "notebook/"
sys.path.append(notebook_dir)

from fonctions.importation import *
from fonctions.fct_preprocess import *

### main :
df = etl(r"..\data\data_raw")

assign_borough(df)
drop_useless_columns(df)
low_modalities_grouping(df)

df.to_csv('..\data\data_clean\data_preprocess.csv', index=False)