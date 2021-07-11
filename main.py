import pandas as pd
from project_functions import *
import sys

def main(filename):
    x_test = pd.read_csv(filename)
    y_test = []
    for decade in x_test.decade.unique().tolist():
        loaded_model = joblib.load(open("models/"+decade + "_model.sav", 'rb'))
        y_temp = loaded_model.predict(x_test[x_test.decade == decade][features])
        y_test += y_temp.tolist()
    pd.DataFrame({id:x_test['id'],"hits":y_test}).to_csv("predicted_y.csv")
    print("predicted values stored to predicted_y.csv")
    return y_test

features = get_features()

filename = sys.argv[1]
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)
