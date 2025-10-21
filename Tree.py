import numpy as np
from Support import *
from evaluation import accuracy


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
        """ Fetch all nodes with two leaves as its child

        Args:
            node (Node) : Node class that are currently being checked
            result_list (List) : List to store nodes with two leaves

        Returns:
            None
        """
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


    def _try_merge(self, parent_node, mergable_node, merged_node, X_val, y_val):
        """ Try to merge nodes with two leaves if accuracy improves

        Args:
            parent_node (Node) : parent of the Node Class that are currently being checked
            mergable_node (Node) : Node class that are currently being checked
            merged_node (Node) : Node class after its two leaves being removed
            X_val (nd.array) : ndarray containing features on validation dataset
            y_val (nd.array) : ndarray containing labels on validation dataset

        Returns:
            bool : True if node is being pruned
        """
        predicted_classes = self.predict(X_val)
        accuracy_pre_merge = accuracy(y_val, predicted_classes)
        print(f"Pre merge accuracy: {accuracy_pre_merge}")

        side = "left" if mergable_node is parent_node.left else "right"
        original = getattr(parent_node, side)
        setattr(parent_node, side, merged_node)

        predicted_classes = self.predict(X_val)
        accuracy_post_merge = accuracy(y_val, predicted_classes)
        print(f"Post merge accuracy: {accuracy_post_merge}")

        if accuracy_post_merge < accuracy_pre_merge:
            print("Reverting merge")
            setattr(parent_node, side, original)
            return False
        return True
    
    def prune_tree(self, X_val, y_val):
        """ Continually prune the decision tree until no further improvements (accuracy) are seen

        Args:
            X_val (nd.array) : ndarray containing features on validation dataset
            y_val (nd.array) : ndarray containing labels on validation dataset

        Returns:
            None
        """
        layer = 0    
        while True:
            num_of_pruned_nodes = 0
            mergable_nodes = []
            self._get_node_with_leaves(result_list=mergable_nodes)

            for mergable_node in mergable_nodes:
                parent_node = mergable_node.parent

                merged_node = Node(
                    value=mergable_node.dominant_y, 
                    depth = mergable_node.depth,
                    parent = parent_node
                )

                if self._try_merge(parent_node, mergable_node, merged_node, X_val, y_val):
                    num_of_pruned_nodes += 1

            print(f"Number of pruned nodes on this layer: {num_of_pruned_nodes}")

            print(f"Number of layer: {layer}")
            print()

            if num_of_pruned_nodes == 0:
                break

            layer += 1
     