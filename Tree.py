import numpy as np
from Support import *


class Node():
    """ Node class that contains information on splitting decision by the tree or contains the final value if it is a terminal node.
    
    """
    def __init__(self, value = None, left = None, right = None, parent = None, threshold = None, feature = None, depth = 0, dominant_y = None):
        self.value = value
        self.depth = depth
        self.left = left
        self.right = right
        self.parent = parent
        self.threshold = threshold
        self.feature = feature
        self.dominant_y = dominant_y

    
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
        self.max_depth = 0


    def fit(self, X, y, depth = 0, parent = None):
        """ Recursively splits the input data based off maximum information gain.

        Args:
            X (nd.array) : ndarray containing features that will be split by decision tree.
            y (nd.array) : ndarray containing the output values that are trying to be classified by decision tree.
            depth (int)  : Integer keeping trach of the depth of the current node.

        Returns:
            Node : Node class that contains information on its value and depth (if terminal node), otherwise contains its depth, child nodes, the threshold
            that it splits, and which feature that threshold belongs to in X
        
        """

        dominant_y = np.argmax(np.bincount(y))

        if len(np.unique(y)) == 1:
            return Node(value = np.argmax(np.bincount(y)), depth = depth, parent = parent)
        
        best_feature_index, best_split_value = find_best_split_feature(X, y)

        left_mask = X[:, best_feature_index] <= best_split_value
        right_mask = X[:, best_feature_index] > best_split_value
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # print(X_left.shape, X_right.shape)
        if(self.max_depth < depth):
            self.max_depth = depth

        #Checks if any split has occured. If none then this is terminal.
        if len(X_left) == len(X):
            ### IMPORTANT(KINDA?): IDK why some of these arrays are like [1, 1, 1, 3, 1, 1, 1]. I don't see why it can't split these from our
            ### code above but I needed to do this otherwise stuck in infinite recursion.
            ### Anyways just something to think about if you read this and remember to delete this comment if we figure it out/before SUBMISSION

            # Returns most frequent value in array 
            return Node(value = np.argmax(np.bincount(y_left)), depth = depth +1, parent = parent)

        if len(X_right) == len(X):
            return Node(value = np.argmax(np.bincount(y_right)), depth = depth+1, parent = parent)

        node = Node(
            depth = depth,
            threshold=best_split_value,
            feature=best_feature_index,
            left = None,
            right = None,
            parent = parent,
            dominant_y = dominant_y
        )

        node.left = self.fit(X_left, y_left, depth = depth + 1, parent=node)
        node.right = self.fit(X_right, y_right, depth = depth + 1, parent=node)

        if depth == 0:
            self.root = node
            print(f"maximum depth: {self.max_depth}")
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
    
    def _get_node_with_leaves(self, node=None, result_list=[]):
        if node is None:
            node = self.root

        # basecase
        if node.left.is_terminal_node() and node.right.is_terminal_node():
            result_list.append(node)
            return

        # recursion body
        if not node.left.is_terminal_node():
            self._get_node_with_leaves(node.left, result_list)

        if not node.right.is_terminal_node():
            self._get_node_with_leaves(node.right, result_list)
    
    def prune_tree(self, X_val, y_val):

        layer = 0    
        while True:
            num_of_pruned_nodes = 0
            mergable_nodes = []
            self._get_node_with_leaves(result_list=mergable_nodes)

            for mergable_node in mergable_nodes:
                predicted_classes = self.predict(X_val)
                acc_pre_merge = np.sum(predicted_classes == y_val)/len(X_val)
                print(f"Pre merge accuracy: {acc_pre_merge}")

                parent_node = mergable_node.parent

                merged_node = Node(
                    value=mergable_node.dominant_y, 
                    depth = mergable_node.depth,
                    parent = parent_node
                )

                node_location = "LEFT" if parent_node.left is mergable_node else "RIGHT"

                if node_location == "LEFT":
                    parent_node.left = merged_node
                else:
                    parent_node.right = merged_node
                
                predicted_classes = self.predict(X_val)
                acc_post_merge = np.sum(predicted_classes == y_val)/len(X_val)
                print(f"Post merge accuracy: {acc_post_merge}")

                if acc_post_merge < acc_pre_merge:
                    print(f"Reverting merge")
                    if node_location == "LEFT":
                        parent_node.left = mergable_node
                    else:
                        parent_node.right = mergable_node
                else:
                    num_of_pruned_nodes += 1

            print(f"Number of pruned nodes on this layer: {num_of_pruned_nodes}")

            print(f"Number of layer: {layer}")
            print()

            if num_of_pruned_nodes == 0:
                break

            layer += 1
     