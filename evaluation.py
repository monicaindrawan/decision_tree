import numpy as np
from Tree import DecisionTree


def evaluate(test_db, trained_tree):
    """
    Calculate the accuracy from validation dataset

    Args:
        test_db : tuple of np.ndarray, shape (n_samples, n_features), (n_samples, 1)
        trained_tree : class, Decision tree node

    Returns:
        accuracy : int
            The model accuracy with regards to the validation data 
    """
    test_data, test_classes = test_db

    predicted_classes = trained_tree.predict(test_data)
    accuracy = np.sum(predicted_classes == test_classes)/len(test_data)
    
    return accuracy



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




def confusion_matrix():
     pass