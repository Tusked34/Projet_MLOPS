import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fonctions.fct_model_evaluation import *

data_predictions = pd.read_csv('data\data_predict\data_predicted.csv')

visualisation_predictions_from_df(data_predictions, output_dir='results')