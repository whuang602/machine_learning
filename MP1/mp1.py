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
    for clss in range(len(n_lst)):
        for line in training_input[i:i+n_lst[clss]]:
            centroids[clss]+= np.array(line)
        centroids[clss][:] = [x/n_lst[clss] for x in centroids[clss]]
        i+=n_lst[clss]
    print("centroids =", centroids)
    
    # [A/B, B/C, C/A]
    vector_slope = [centroids[0]-centroids[1],centroids[1]-centroids[2],centroids[2]-centroids[0]]
    print("slope =",vector_slope)
    
    # Halfway between each pair of centroids
    halfway = [(centroids[0]+centroids[1])/2,(centroids[1]+centroids[2])/2,(centroids[2]+centroids[0])/2]
    print("halfway point =", halfway)
    
    # Find Perpendicular Bisector. Ex PB = midway_point + t * ortho_vector
    # orth = []
    # for slope in vector_slope:
    #     if slope.any() == 0:
    #         if slope[0] == 0:
    #             orth.append(np.array([1,0,0]))
    #             continue
    #         if slope[1] == 0:
    #             orth.append(np.array([0,1,0]))
    #             continue
    #         orth.append(np.array([0,0,1]))
    #     else:
    #         orth.append(np.array([0,1,-slope[1]/slope[2]]))
    # print("orthogonal =", orth)
    
    # Or just find the orthogonal plane containing the middle point
    
    # a(x-x_0) + b(y-y_0) + c(z-z_0) = ax + by + cz - d
    # slope[0] * x + slope[1] * y + slope[2] * z = slope[0] * halfway[0][0] + slope[1] * halfway[0][1] + slope[2] * halfway[0][2]
    d_offsets = []
    for i in range(len(n_lst)):
        d_offsets.append(vector_slope[i][0] * halfway[i][0] + vector_slope[i][1] * halfway[i][1] + vector_slope[i][2] * halfway[i][2])
    print("Offset =", d_offsets)
    
    true_class, other_class, positive_side = ["A","B","C"], ["B", "C", "A"], []
    for i in range(len(n_lst)):
        # equivalent to vector_slope[index][0] * centroids[index][0] + vector_slope[index][1] * centroids[index][1] + vector_slope[index][2] * centroids[index][2] - d
        positive_side.append(true_class[i]) if (np.sum(vector_slope[i] * centroids[i]) - d_offsets[i]) > 0 else positive_side.append(other_class[i])
    print(positive_side)
    
    # --------------------------------------- Testing ---------------------------------------
    
    
    d = testing_input[0][0]
    n_lst = []
    for i in testing_input[0][1:]:
        n_lst.append(i)
    
    i, index = 1, 0
    margin = [[0,0],[0,0],[0,0]]
    for clss in range(len(n_lst)):
        for line in testing_input[i:i+n_lst[clss]]:
            if (np.sum(vector_slope[index] * line) - d_offsets[index]) > 0:
                if positive_side[index] == true_class[index]:
                    margin[index][0]+=1
                else:
                    margin[index][1]+=1
            else:
                if positive_side[index] == true_class[index]:
                    margin[index][1] +=1
                else:
                    margin[index][0] +=1
        i+=n_lst[clss]
        index+=1
    print(margin)
    
    
        
    return results