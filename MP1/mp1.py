# Starter code for CS 165B MP1 Fall 2023
import numpy as np
def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """

    results = {"tpr":0,"fpr":0,"error_rate":0,"accuracy":0,"precision":0}
    
    d = training_input[0][0]
    n_lst = []
    for i in training_input[0][1:]:
        n_lst.append(i)
    
    # [A,B,C] or potentially more
    centroids = []
    for i in range(len(n_lst)):
        centroids.append(np.array([0.0,0.0,0.0])) # dependent on var d
        
    i = 1 # temp var, end point for each class, accounts for the number of data in each class.
    # Ex. skip 1, up til but excl. 501, up til but excl. 1001, up til but excl. 1501
    for cl in range(len(n_lst)):
        for line in training_input[i:i+n_lst[cl]]:
            centroids[cl]+= np.array(line)
        centroids[cl][:] = [x/n_lst[cl] for x in centroids[cl]]
        i+=n_lst[cl]
    print(centroids)
    
    # [A/B, B/C, C/A]
    vector_slope = [centroids[0]-centroids[1],centroids[1]-centroids[2],centroids[2]-centroids[0]]
    halfway = [(centroids[0]+centroids[1])/2,(centroids[1]+centroids[2])/2,(centroids[2]+centroids[0])/2]
    print(vector_slope)
    print(halfway)
    
    
    
        
    return results