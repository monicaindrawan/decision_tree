from main import *
import numpy as np

def evaluate(test_db, trained_tree=None):
    test_data, test_classes = test_db

    predicted_classes = trained_tree.predict(test_data)
    accuracy = np.sum(predicted_classes == test_classes)/len(test_data)
    return accuracy
  


def run_simulation(test_db, trained_tree):
    folds = np.array_split(test_db, 10)
    accuracy = []
    for i in range(len(folds)):
            test_fold = folds[i]
            train_folds = np.vstack(folds[:i] + folds[i+1:])
            trained_tree_for_folds = trained_tree.fit(train_folds)
            accuracy.append(evaluate(test_fold, trained_tree_for_folds))
    average_accuracy = np.mean(accuracy)
    return average_accuracy

def confusion_matrix():
     pass