
import pandas as pd
from project_functions import *
from sklearn.neural_network import MLPClassifier

spotify_data= pd.read_csv("wm_project.csv")

# Removing identification data from the train and test
spotify_data = spotify_data.drop(['uri','track'],axis = 1).set_index('id')

features = get_features()

score_train_test_val(spotify_data, features, MLPClassifier(alpha = 0.0001))