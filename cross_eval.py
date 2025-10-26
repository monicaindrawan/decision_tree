import numpy as np
from tree import DecisionTree
from evaluation import *

# For reproducibility
random_generator = np.random.default_rng(seed = 42)

def cross_validation(X, y, k=10):
    """
    Calculate the accuracy from validation dataset using 10 cross validation approach

    Args:
        X_data : np.ndarray, shape (n_samples, n_features)
        y_data : np.ndarray, shape (n_samples, 1)

    Returns:
        average_accuracy : int
            The average accuracy with regards to the validation data
    """

    confusion_matrix_list = []
    count = 0

    for train_indices, test_indices in train_test_k_fold(k, len(X), random_generator):
        x_train = X[train_indices, :]
        y_train = y[train_indices]
        x_test = X[test_indices, :]
        y_test = y[test_indices]

        tree = DecisionTree()

        tree.fit(x_train, y_train)
        predictions = tree.predict(x_test)
        confusion_matrix_list.append(confusion_matrix(y_test, predictions))
        #print(confusion_matrix_list)
        count += 1
        print("Iteration:", f"{count}/{k}", end='\r')

    average_confusion_matrix = np.sum(confusion_matrix_list, axis=0)/(k)
    average_accuracy = accuracy_from_confusion(average_confusion_matrix)
    average_precision = precision_from_confusion(average_confusion_matrix)
    average_recall = recall_from_confusion(average_confusion_matrix)
    average_f1 = f1_score(average_precision[0], average_recall[0])


    # print(f"confusion matrix: \n {average_confusion_matrix}")
    # print(f"average accuracy: {average_accuracy}")  
    # print(f"average precision per class: {average_precision[0]}")
    # print(f"macro average precision: {average_precision[1]}")
    # print(f"average recall per class: {average_recall[0]}")
    # print(f"macro average recall: {average_recall[1]}")
    # print(f"average F1 per class: {average_f1[0]}")
    # print(f"macro average F1: {average_f1[1]}")

    return (average_accuracy, average_precision[1], average_recall[1], average_f1[1])



def nested_cross_validation(X, y):
    n_outer_folds = 10
    n_inner_folds = 9
    count = 0
    confusion_matrix_list = []
    total_max_depth_before = 0
    total_max_depth_after = 0

    for i, (trainval_indices, test_indices) in enumerate(train_test_k_fold(n_outer_folds, len(X), random_generator)):
        x_trainval = X[trainval_indices, :]
        y_trainval = y[trainval_indices]
        x_test = X[test_indices, :]
        y_test = y[test_indices]

        splits = train_test_k_fold(n_inner_folds, len(x_trainval), random_generator)

        for (train_indices, val_indices) in splits:
            x_train = x_trainval[train_indices, :]
            y_train = y_trainval[train_indices]
            x_val = x_trainval[val_indices, :]
            y_val = y_trainval[val_indices]

            tree = DecisionTree()
            tree.fit(x_train, y_train)

            max_depth_before, max_depth_after = tree.prune_tree(x_val, y_val)
            total_max_depth_before += max_depth_before
            total_max_depth_after += max_depth_after
            
            
            predictions = tree.predict(x_test)
            confusion_matrix_list.append(confusion_matrix(y_test, predictions))
            count += 1
            print("Iteration:", f"{count}/{n_outer_folds * n_inner_folds}", end='\r')


    average_confusion_matrix = np.sum(confusion_matrix_list, axis=0)/(n_outer_folds*n_inner_folds)
    average_accuracy = accuracy_from_confusion(average_confusion_matrix)
    average_precision = precision_from_confusion(average_confusion_matrix)
    average_recall = recall_from_confusion(average_confusion_matrix)
    average_f1 = f1_score(average_precision[0], average_recall[0])

    average_max_depth_before = total_max_depth_before / count
    average_max_depth_after = total_max_depth_after / count


    # print(f"confusion matrix: \n {average_confusion_matrix}")
    # print(f"average accuracy: {average_accuracy}")  
    # print(f"average precision per class: {average_precision[0]}")
    # print(f"macro average precision: {average_precision[1]}")
    # print(f"average recall per class: {average_recall[0]}")
    # print(f"macro average recall: {average_recall[1]}")
    # print(f"average F1 per class: {average_f1[0]}")
    # print(f"macro average F1: {average_f1[1]}")
    # print(f"average max depth before pruning: {average_max_depth_before}")
    # print(f"average max depth before pruning: {average_max_depth_after}")

    return (average_accuracy, average_precision[1], average_recall[1], average_f1[1], average_max_depth_before, average_max_depth_after)
