# Starter code for CS 165B MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    pd.options.mode.chained_assignment = None
    X = (training_data.drop(columns=["target","QUANTIZED_WORK_YEAR","QUANTIZED_INC","QUANTIZED_AGE"])).copy()
    Y = training_data["target"]
    print(X.shape)

    X_test = testing_data.copy()
    X_test = X_test.drop(columns=["QUANTIZED_WORK_YEAR","QUANTIZED_INC","QUANTIZED_AGE"])
    # preprocessing
    categorical_features = ["NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE"]
    # removed categorical features: "QUANTIZED_WORK_YEAR","QUANTIZED_INC","QUANTIZED_AGE"
    for col in categorical_features: # label encoding babyyy
        mp = {}
        counter = 0
        for i in range(len(X[col])):
            if X[col][i] not in mp:
                mp[X[col][i]] = counter
                counter+=1
            X[col][i] = mp[X[col][i]]
        for i in range(len(X_test[col])):
            if X_test[col][i] not in mp:
                mp[X_test[col][i]] = counter
                counter+=1
            X_test[col][i] = mp[X_test[col][i]]
        
            
    
    # binary_features = ["CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","FLAG_MOBIL","FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL"]
    # continuous_features = ["CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","CNT_FAM_MEMBERS","DAYS_EMPLOYED"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(30,132), max_iter=1000, random_state=42) #0.564 F1 score
    
    model.fit(X, Y)
    
    predict = model.predict(X_test)
    print(predict)

    return predict


if __name__ == '__main__':

    training = pd.read_csv('./data/train.csv')
    development = pd.read_csv('./data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)

    


    


