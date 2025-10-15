import numpy as np
from Support import *

class Node():
    """ Node class that contains information on splitting decision by the tree or contains the final value if it is a terminal node.
    
    """
    def __init__(self, value = None, left = None, right = None, threshold = None, feature = None, depth = 0):
        self.value = value
        self.depth = depth
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
    
    def is_terminal_node(self):
        if self.value is None:
            return False
        else:
            return True


class DecisionTree():
    """ DecisionTree class that contains the root node and is used to create decision tree (with fit()) and to predict using that tree (with predict())
    """
    def __init__(self):
        self.root = None


    def fit(self, X, y, depth = 0):
        """ Recursively splits the input data based off maximum information gain.

        Args:
            X (nd.array) : ndarray containing features that will be split by decision tree.
            y (nd.array) : ndarray containing the output values that are trying to be classified by decision tree.
            depth (int)  : Integer keeping trach of the depth of the current node.

        Returns:
            Node : Node class that contains information on its value and depth (if terminal node), otherwise contains its depth, child nodes, the threshold
            that it splits, and which feature that threshold belongs to in X
        
        """
        if len(np.unique(y)) == 1:
            return Node(value = np.argmax(np.bincount(y)), depth = depth)
        
        best_feature_index, best_split_value = find_best_split_feature(X, y)

        left_mask = X[:, best_feature_index] <= best_split_value
        right_mask = X[:, best_feature_index] > best_split_value
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        #Checks if any split has occured. If none then this is terminal.
        if len(X_left) == len(X):
            ### IMPORTANT(KINDA?): IDK why some of these arrays are like [1, 1, 1, 3, 1, 1, 1]. I don't see why it can't split these from our
            ### code above but I needed to do this otherwise stuck in infinite recursion.
            ### Anyways just something to think about if you read this and remember to delete this comment if we figure it out/before SUBMISSION

            # Returns most frequent value in array 
            return Node(value = np.argmax(np.bincount(y_left)), depth = depth +1)

        if len(X_right) == len(X):
            return Node(value = np.argmax(np.bincount(y_right)), depth = depth+1)

        left_subtree = self.fit(X_left, y_left, depth = depth + 1)
        right_subtree = self.fit(X_right, y_right, depth = depth + 1)
        
        node = Node(
            depth = depth,
            threshold=best_split_value,
            feature=best_feature_index,
            left = left_subtree,
            right = right_subtree
        )

        if depth == 0:
            self.root = node
        return node
    

    def _traverse_tree(self, x, node=None):
        """ Traverses the tree based on information from x (1-D array) and outputs value of terminal node reached.

        Args:
            x (1-D ndarray) : feature space that is being predicted.
            node (Node) : Node class that x is currently being checked against in the decision tree.

        Returns:
            int : Value of terminal node based off input data x
        """
        if node is None:
            node = self.root
        if node.is_terminal_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    

    def predict(self, X):
        return np.array([self._traverse_tree(row) for row in X])


