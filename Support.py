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

def find_ordered_value_changes(X, y):
    """ Identify points where the target variable y changes when the dataset
        is sorted by each feature column in X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix where each column represents a feature.

    y : np.ndarray, shape (n_samples,) 
        Target vector or integer class labels corresponding to each row of `X`.

    Returns
    -------
    change_points_list : list of list of int
        For each feature, a list of indices (in the sorted order)
        indicating where `y` changes when sorted by that feature.

    sorted_x_list : list of np.ndarray
        List of feature-sorted `X` arrays, one per feature column.

    sorted_y_list : list of np.ndarray
        List of corresponding `y` arrays sorted by each feature column.
    """
    
    output = []
    sorted_y_list = []
    sorted_x_list = []
    for index in range(X.shape[1]):
        sorted_indices = np.argsort(X[:, index])
        y_sorted = y[sorted_indices]
        y_current = y_sorted[0]
        change_index = []
        for i in range(1, len(y_sorted)):
            if y_sorted[i] != y_current:
                change_index.append(i)
            y_current = y_sorted[i]
        output.append(change_index)
        sorted_y_list.append(y_sorted)
        sorted_x_list.append(X[sorted_indices])
    return output, sorted_x_list, sorted_y_list



def decision_tree_learning(X, y, depth):
    pass
