from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import json
import xgboost
import joblib

def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def get_features():
    '''

    :return: features added after feature selection can be readily used for training or predicting values
    '''
    features = ['danceability',
                'energy',
                'mode',
                'valence',
                'acousticness',
                'speechiness',
                'instrumentalness',
                'liveness',
                'tempo',
                'chorus_hit']
    return features

def score_train_test_val(data,features, model):
    '''
    :param  features : variables that the model would utilize for training
    :param  model    : binary classification model
    :return roc_auc_score    : training data
    :return roc_auc_score    : validation data
    :return confusion matrix : validation data

    '''
    Y_Train_True = []
    Y_Train_Predicted = []
    Y_Validation_True = []
    Y_Validation_Predicted = []
    for i in data.decade.unique().tolist():
        print(i)
        segment_data = data[data.hit.notna()][data.decade == i].copy()
        x = segment_data[features]
        y = segment_data["hit"]
        x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.33)

        if (type(model) != xgboost.sklearn.XGBClassifier):
            clf = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                            ('normalizer',
                             preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)),
                            ('model', model)]).fit(x_train, y_train)
            filename = i + '_model.sav'
            joblib.dump(clf, filename)
            print("model " + i + " saved")

        if (type(model) == xgboost.sklearn.XGBClassifier):
            clf = model.fit(x_train, y_train)
            clf.save_model(i + '_model.json')
            print("model " + i + " saved")

        y_predicted_val = clf.predict(x_validate)
        y_predicted_train = clf.predict(x_train)

        Y_Validation_Predicted += y_predicted_val.tolist()
        Y_Train_Predicted += y_predicted_train.tolist()
        Y_Train_True += y_train.tolist()
        Y_Validation_True += y_validate.tolist()
    print(roc_auc_score(Y_Train_True, Y_Train_Predicted))
    print(roc_auc_score(Y_Validation_True,Y_Validation_Predicted))

    return roc_auc_score(Y_Train_True, Y_Train_Predicted), roc_auc_score(Y_Validation_True,Y_Validation_Predicted), confusion_matrix(Y_Validation_True, Y_Validation_Predicted)
