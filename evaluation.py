import numpy as np
from Tree import DecisionTree

def evaluate(test_db, trained_tree):
    test_data, test_classes = test_db

    predicted_classes = trained_tree.predict(test_data)
    accuracy = np.sum(predicted_classes == test_classes)/len(test_data)
    
    return accuracy
  


def cross_validation(X_data, y_data, k=10):
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