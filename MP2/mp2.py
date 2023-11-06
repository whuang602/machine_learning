# CS 165B MP2
import math
import random
import numpy as np
import pandas as pd

from typing import List

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """

    Inputs:
        training_data: pd.DataFrame
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    
    # Set-up
    training = [] # x_i
    label = [] # y_i
    for id, row in training_data.iterrows():
        x = row.to_numpy()
        # label.append(int(x[-1]))
        label.append(1 if int(x[-1]) > 0 else -1)
        x[-1] = 1
        training.append(x)
    
    learning_rate = 0.01
    
    # Brainstorming:
    # SGD mini-batch
    # S(x) = wx ---- (offset b is constained in w (ex. w_(n+1)))
    # y^ = sign(wx)
    # given x1-xn, set x_(n+1) to 1
    # w will have w1-w_(n+1) elements
    # Logistic loss log2(1+exp(-y*S(x))) -> (-exp(-y*S(x)))*(-yS(x))' / (1+exp(-y*S(x)))
    # (-yS(x))' = -yS'(x) = -yx 
    # exp = np.exp(x)
    
    def SGD(y, x, batch, w, learning_rate) -> np.array:
        update = np.array([0.0 for i in range(len(w))])
        for i in batch:
            curr_update = (-y[i]*x[i])*(np.exp(-y[i]*np.dot(x[i][0:-1],w[0:-1]))/ (1+np.exp(-y[i]*np.dot(x[i][0:-1],w[0:-1]))))
            curr_update[-1] = (-y[i])*(np.exp(-y[i]*np.dot(x[i][-1],w[-1]))/ (1+np.exp(-y[i]*np.dot(x[i][-1],w[-1])))) #??????
            update+=curr_update
        return w - learning_rate * (update/len(batch))
    
    # randomize w, gives a fluctuating F1 score
    # w = np.random.rand(len(training[0]))
    
    # initiate w to have all 1s, works better than 0s (presumably as the optimal w is closer to 1)
    w = np.array([1 for i in range(len(training[0]))])
    
    for epoch in range(2000):
        batch = np.random.choice(len(label), size=30, replace=False)
        w = SGD(label, training, batch, w, learning_rate)
    
    print("w =", w)
    
    # Given post-trained w, test on testing_data
    y_hat = np.array([])
    for id, row in testing_data.iterrows():
        x = row.to_numpy()
        x = np.append(x,-1) # subtract the bias
        y_hat = np.append(y_hat, 1 if np.sign(np.dot(x,w)) >= 0 else 0)
    print("prediction =", y_hat)

    # return np.array containing predicted labels y_hat for the testing data
    return y_hat

if __name__ == '__main__':
    # load data
    training = pd.read_csv('data/train.csv')
    testing = pd.read_csv('data/dev.csv')
    target_label = testing['target']
    testing.drop('target', axis=1, inplace=True)

    # run training and testing
    prediction = run_train_test(training, testing)

    # Example metric 1: check accuracy 
    target_label = target_label.values
    print("Dev Accuracy: ", np.sum(prediction == target_label) / len(target_label))
    
    # Metric 2: F1 score
    #F1 = (2*TP)/(P+P^)
    
    TP, P, P_hat = 0, 0, 0
    for i in range(len(target_label)):
        if target_label[i] == 1:
            P+=1
        if prediction[i] == 1:
            P_hat+=1
        if target_label[i] == 1 and prediction[i] == 1:
            TP+=1
            
    F1 = (2*TP)/(P+P_hat)
    print("F1: ", F1)
    


    


