from tree import DecisionTree
from evaluation import *
from cross_eval import cross_validation, nested_cross_validation
from support import load_data


if __name__ == "__main__":
    X_noisy, y_noisy = load_data("wifi_db/noisy_dataset.txt")
    X_clean, y_clean = load_data("wifi_db/clean_dataset.txt")

    # Run this for cross validation on unpruned tree
    cross_validation(X_noisy, y_noisy)
    cross_validation(X_clean, y_clean)

    # Run this for nested cross validation on pruned tree
    nested_cross_validation(X_noisy, y_noisy)
    nested_cross_validation(X_clean, y_clean)
