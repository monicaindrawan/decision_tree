import numpy as np



def entropy(y):
    s = 0
    for class_label in np.unique(y):
        p = np.sum(class_label == y) / len(y)
        s -= p * np.log2(p)

    return s


def find_ordered_value_changes(X, y, index):
    """
    Identify positions where the target variable y changes when the dataset
    is sorted by a single feature column in X.

    Args:
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix containing the input data.

        y : np.ndarray, shape (n_samples,)
            Target vector or class labels corresponding to each sample in `X`.

        index : int
            The column index of `X` to use for sorting and analysis.

    Returns:
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



def find_ideal_split(x_sorted, y_sorted, splits):
    """
    Finds the optimal split condition for a feature (x) based on maximizing information gain
    for the target variable (y).

    Args:
        x_sorted (np.array): A list or array of **sorted** feature values.
        y_sorted (np.array): A list or array of corresponding target/class
                                     labels, **sorted** according to the order in `x_sorted`.
        splits (np.array): A list of integer indices where a split can occur.
                                   These indices indicate the point *after* which the
                                   data is split (i.e., the first element of the right child).

    Returns:
        tuple(float): The feature value (split point) that yields the maximum information gain.
               This is calculated as the midpoint between the last element of the left
               split and the first element of the right split.
    """
       
    # calculate total entropy
    total_entropy = entropy(y_sorted)

    split_value = 0
    max_gain = 0

    for s in splits:
        # calculate the entropy of each split
        left_list = y_sorted[:s]
        right_list = y_sorted[s:]
        left_entropy = entropy(left_list)
        right_entropy = entropy(right_list)

        # calculate the weighted entropy and the gain
        weighted_entropy = len(left_list) / len(y_sorted) * left_entropy + len(right_list) / len(y_sorted) * right_entropy
        gain = total_entropy - weighted_entropy

        # if the gain is bigger than the previously calculated gain, find the split condition
        if gain > max_gain:
            split_value = (x_sorted[s-1] + x_sorted[s]) / 2
            max_gain = gain

    return split_value, max_gain



def find_best_split_feature(X, y):
    max_gain = 0
    best_split_value = 0
    best_feature_index = 0

    for feature_index in range(X.shape[1]):
        splits, x_sorted, y_sorted = find_ordered_value_changes(X, y, feature_index)

        feature_split_value, feature_gain = find_ideal_split(x_sorted[:,feature_index], y_sorted, splits)

        if feature_index == 0:
            best_split_value = feature_split_value

        #print(feature_index, feature_split_value, feature_gain)

        if feature_gain > max_gain:
            best_split_value = feature_split_value
            max_gain = feature_gain
            best_feature_index = feature_index

    return best_feature_index, best_split_value