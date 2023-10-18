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
    
    d = training_input[0][0]
    n_lst = [i for i in training_input[0][1:]]
    
    # [A,B,C] or potentially more
    centroids = [np.array([0.0 for x in range(d)]) for i in range(len(n_lst))] # dependent on var d
        
    i = 1 # temp var, end point for each class, accounts for the number of data in each class.
    # Ex. skip 1, up til but excl. 501, up til but excl. 1001, up til but excl. 1501
    for clss in range(len(n_lst)):
        for line in training_input[i:i+n_lst[clss]]:
            centroids[clss]+= np.array(line)
        centroids[clss][:] = [x/n_lst[clss] for x in centroids[clss]]
        i+=n_lst[clss]
    print("centroids =", centroids)
    
    # [A/B, B/C, C/A]
    vector_slope = [centroids[0]-centroids[1],centroids[1]-centroids[2],centroids[0]-centroids[2]]
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
    # print(positive_side)
    
    # --------------------------------------- Testing ---------------------------------------
    
    
    d = testing_input[0][0]
    n_lst = [ i for i in testing_input[0][1:] ]
    
    i, index = 1, 0
    # vector_slope: [ A/B , B/C, C/A]
    final_results = []
    for clss in range(len(n_lst)):
        for line in testing_input[i:i+n_lst[clss]]:
            curr_index = 0
            calc = np.sum(vector_slope[curr_index] * line) - d_offsets[curr_index]
            # curr_index = 2 if (calc > 0 and positive_side[curr_index] == "A") or (calc < 0 and positive_side[curr_index] != "A") else 1
            if calc > 0: # positive side of A/B
                if positive_side[0] == "A": # point is A, now check A or C
                    calc = np.sum(vector_slope[2] * line) - d_offsets[2]
                    final_results.append("A") if ( calc >= 0 and positive_side[2] =="A" ) or (calc < 0 and positive_side[2] != "A") else final_results.append("C")
                else: # point is B, now check B or C
                    calc = np.sum(vector_slope[1] * line) - d_offsets[1]
                    final_results.append("B") if ( calc >= 0 and positive_side[1] =="B" ) or (calc < 0 and positive_side[1] != "B") else final_results.append("C")
                continue      
            if calc < 0: # negative side of A/B
                if positive_side[0] == "A": # point is B
                    calc = np.sum(vector_slope[1] * line) - d_offsets[1]
                    final_results.append("B") if ( calc >= 0 and positive_side[1] =="B" ) or (calc < 0 and positive_side[1] != "B") else final_results.append("C")
                else: # point is A
                    calc = np.sum(vector_slope[2] * line) - d_offsets[2]
                    final_results.append("A") if ( calc >= 0 and positive_side[2] =="A" ) or (calc < 0 and positive_side[2] != "A") else final_results.append("C")
                continue
            # on the A/B discrimnant function plane (tie), prefer A -> test A or C 
            calc = np.sum(vector_slope[2] * line) - d_offsets[2]
            final_results.append("A") if ( calc >= 0 and positive_side[2] =="A" ) or (calc < 0 and positive_side[2] != "A") else final_results.append("C")
        i+=n_lst[clss]
        index+=1
    
    print(final_results)
    
    confusion_matrix = [[0,0,0] for x in range(3)]
    convenience = {"A":0, "B":1, "C":2}
    i, index = 1, 0
    for clss in range(len(n_lst)):
        for result in final_results[i:i+n_lst[clss]]:
            confusion_matrix[convenience[result]][convenience[true_class[index]]]+=1 # imagine writing nested switch statements, yikes
        i+=n_lst[clss]
        index+=1
    for i in range(3):
        print(confusion_matrix[i])
    
    # [True Positive, False Positive, False Negative, True Negative]
    confusion_matrix_specific = [[0.0,0.0,0.0,0.0] for x in range(3)]
    
    # Class A
    confusion_matrix_specific[0][0]+=confusion_matrix[0][0]
    confusion_matrix_specific[0][1]+=confusion_matrix[0][1]+confusion_matrix[0][2]
    confusion_matrix_specific[0][2]+=confusion_matrix[1][0]+confusion_matrix[2][0]
    confusion_matrix_specific[0][3]+=confusion_matrix[1][1]+confusion_matrix[1][2]+confusion_matrix[2][1]+confusion_matrix[2][2]
    
    # Class B
    confusion_matrix_specific[1][0]+=confusion_matrix[1][1]
    confusion_matrix_specific[1][1]+=confusion_matrix[1][0]+confusion_matrix[1][2]
    confusion_matrix_specific[1][2]+=confusion_matrix[0][1]+confusion_matrix[2][1]
    confusion_matrix_specific[1][3]+=confusion_matrix[0][0]+confusion_matrix[0][2]+confusion_matrix[2][0]+confusion_matrix[2][2]
    
    # Class C
    confusion_matrix_specific[2][0]+=confusion_matrix[2][2]
    confusion_matrix_specific[2][1]+=confusion_matrix[2][0]+confusion_matrix[2][1]
    confusion_matrix_specific[2][2]+=confusion_matrix[0][2]+confusion_matrix[1][2]
    confusion_matrix_specific[2][3]+=confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]
    print(confusion_matrix_specific)
            
    pre_average_results = [ np.array([0.0,0.0,0.0,0.0,0.0]) for x in range(3)]
    
    index=0
    for clss in confusion_matrix_specific:
        pre_average_results[index][0] = clss[0] / (clss[0] + clss[2])
        pre_average_results[index][1] = clss[1] / (clss[1] + clss[3])
        pre_average_results[index][2] = (clss[1] + clss[2]) / np.sum(clss)
        pre_average_results[index][3] = (clss[0] + clss[3]) / np.sum(clss)
        pre_average_results[index][4] = clss[0] / (clss[0] + clss[1])
        index+=1
    
    # tpr = TP / P where P = TP + FN
    # fpr = FP / N where N = FP + TN
    # error rate = (FP + FN) / (P+N)
    # accuracy = (TP+TN)/(P+N)
    # Precision = TP / P_hat where P_hat = TP + FP
    # [TP, FP, FN, TN]
    results = {"tpr": (pre_average_results[0][0]+pre_average_results[1][0]+pre_average_results[2][0])/3,
               "fpr": (pre_average_results[0][1]+pre_average_results[1][1]+pre_average_results[2][1])/3,
               "error_rate": (pre_average_results[0][2]+pre_average_results[1][2]+pre_average_results[2][2])/3,
               "accuracy": (pre_average_results[0][3]+pre_average_results[1][3]+pre_average_results[2][3])/3,
               "precision": (pre_average_results[0][4]+pre_average_results[1][4]+pre_average_results[2][4])/3
               }
    # results = {"tpr": 0.8933333333333334,
    #            "fpr": 0.05333333333333334,
    #            "error_rate": 0.07111111111111111,
    #            "accuracy": 0.9288888888888889,
    #            "precision": 0.8923242082662372}
    print(results)
    
    return results