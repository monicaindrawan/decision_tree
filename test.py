from support import load_data, train_test_split
from tree import DecisionTree
# IMPORTANT: Make sure matplotlib is available, if not, do not includes this module
from visualisation import plot_tree
from evaluation import *


if __name__ == "__main__":
    # Load the data
    X, y = load_data("wifi_db/noisy_dataset.txt")     # Replace load_data argument with the path of any data.txt files

    # Split the dataset for training and testing
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y)

    # Construct and fit the decision tree classifier
    decision_tree_classifier = DecisionTree()
    decision_tree_classifier.fit(X_train, y_train)
    
    # Prune the tree
    decision_tree_classifier.prune_tree(X_val, y_val)

    # Predict the test dataset
    y_prediction = decision_tree_classifier.predict(X_test)

    # Show the Confusion Matrix
    print(confusion_matrix(y_test, y_prediction))

    # IMPORTANT: Make sure matplotlib is available
    # Visualise the pruned tree
    plot_tree(decision_tree_classifier)