import numpy as np
from Tree import DecisionTree
from evaluation import *


def cross_validation(X_data, y_data, k=10):
    """
    Calculate the accuracy from validation dataset using 10 cross validation approach

    Args:
        X_data : np.ndarray, shape (n_samples, n_features)
        y_data : np.ndarray, shape (n_samples, 1)

    Returns:
        average_accuracy : int
            The average accuracy with regards to the validation data
    """
    X_folds = np.array_split(X_data, k)
    y_folds = np.array_split(y_data, k)

    scores = []

    for i in range(k):
        X_test_fold = X_folds[i]
        y_test_fold = y_folds[i]

        X_train_fold = np.vstack(X_folds[:i] + X_folds[i+1:])
        y_train_fold = np.hstack(y_folds[:i] + y_folds[i+1:])

        tree = DecisionTree()

        tree.fit(X_train_fold, y_train_fold)

        score = evaluate((X_test_fold,y_test_fold), tree)
        scores.append(score)
    
    average_accuracy = np.mean(scores)
    return average_accuracy


def nested_cross_validation(X, y):
    n_outer_folds = 10
    n_inner_folds = 9
    running_sum_accuracy = 0
    count = 0
    confusion_matrix_list = []

    for i, (trainval_indices, test_indices) in enumerate(train_test_k_fold(n_outer_folds, len(X), random_generator = np.random.default_rng())):
        x_trainval = X[trainval_indices, :]
        y_trainval = y[trainval_indices]
        x_test = X[test_indices, :]
        y_test = y[test_indices]

        splits = train_test_k_fold(n_inner_folds, len(x_trainval))

        for (train_indices, val_indices) in splits:
            x_train = x_trainval[train_indices, :]
            y_train = y_trainval[train_indices]
            x_val = x_trainval[val_indices, :]
            y_val = y_trainval[val_indices]

            tree = DecisionTree()
            tree.fit(x_train, y_train)
            tree.prune_tree(x_val, y_val)
            #accuracy_list.append(evaluate((x_test, y_test), tree))
            predictions = tree.predict(x_test)
            confusion_matrix_list.append(confusion_matrix(y_test, predictions))
            #y_predictions = tree.predict(x_test)
            #accuracy = accuracy(y_test, y_predictions)
            #running_sum_accuracy += accuracy
            count += 1
            print(count)

    average_confusion_matrix = np.sum(confusion_matrix_list, axis=0)/(n_outer_folds*n_inner_folds)
    average_accuracy = accuracy_from_confusion(average_confusion_matrix)
    average_precision = precision_from_confusion(average_confusion_matrix)
    average_recall = recall_from_confusion(average_confusion_matrix)
    average_f1 = f1_score(average_precision[0], average_recall[0])


    print(f"confusion matrix: \n {average_confusion_matrix}")
    print(f"average accuracy: {average_accuracy}")  
    print(f"average precision per class: {average_precision[0]}")
    print(f"macro average precision: {average_precision[1]}")
    print(f"average recall per class: {average_recall[0]}")
    print(f"macro average recall: {average_recall[1]}")
    print(f"average F1 per class: {average_f1[0]}")
    print(f"macro average F1: {average_f1[1]}")
