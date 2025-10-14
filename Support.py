import numpy as np


class BinaryTreeNode:
    def __init__(self, value, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value


def entropy(y):
    s = 0
    for class_label in np.unique(y):
        p = np.sum(class_label == y) / len(y)
        s -= p * np.log2(p)

    return s

def information_gain(S_all, S_left, S_right):
    pass


def find_split(X, y):
    pass

def find_ordered_value_changes(X, y, index):
    """
    Identify positions where the target variable y changes when the dataset
    is sorted by a single feature column in X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix containing the input data.

    y : np.ndarray, shape (n_samples,)
        Target vector or class labels corresponding to each sample in `X`.

    index : int
        The column index of `X` to use for sorting and analysis.

    Returns
    -------
    change_indices : list of int
        Indices (in sorted order) where the value of y changes
        as X[:, index] increases.  

    X_sorted : np.ndarray, shape (n_samples, n_features)
        The feature matrix sorted by column `index`.

    y_sorted : np.ndarray, shape (n_samples,)
        The target vector sorted according to column `index`.
    """
    
    sorted_indices = np.argsort(X[:, index])
    y_sorted = y[sorted_indices]
    y_current = y_sorted[0]
    change_index = []
    for i in range(1, len(y_sorted)):
        if y_sorted[i] != y_current:
            change_index.append(i)
        y_current = y_sorted[i]
    return change_index, X[sorted_indices], y_sorted



def decision_tree_learning(X, y, depth):
    pass
