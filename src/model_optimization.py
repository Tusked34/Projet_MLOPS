import sys
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fonctions.fct_model_optimization import *
from fonctions.fct_model_training import *

data = pd.read_csv("data\data_clean\data_preprocess.csv")
X, y, num_features, catg_features = data_split(df= data)
transformer = preprocess_data(features = X, target = y, numeric_features = num_features, categorical_features = catg_features)

print("\nDébut du processus d'hyperparameters-tuning")
optimisation_hyperparametres(features = X, target = y, preprocessor = transformer, 
                             output_path_prediction='data\data_predict')