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
    # For each feature in X
    for feature_index in range(len(X)):
        sorted_indices = np.argsort(X[feature_index])
        X, y = X[sorted_indices], y[sorted_indices]

        


def decision_tree_learning(X, y, depth):
    pass
