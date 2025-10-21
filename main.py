import numpy as np
from Tree import DecisionTree
from evaluation import *
from cross_eval import nested_cross_validation

def load_data(file):
    data = np.loadtxt(file)

    x = data[:, :-1]
    y = data[:, -1].astype(int)

    return x, y

X_noisy, y_noisy = load_data("wifi_db/noisy_dataset.txt")
X_clean, y_clean = load_data("wifi_db/clean_dataset.txt")


tree = DecisionTree()

X_train_noisy, y_train_noisy = (X_noisy[20:], y_noisy[20:])
X_val_noisy, y_val_noisy = (X_noisy[:20], y_noisy[:20])
val_db = (X_val_noisy, y_val_noisy)
tree.fit(X_train_noisy, y_train_noisy)

# X_train_clean, y_train_clean = (X_clean[20:], y_clean[20:])
# X_val_clean, y_val_clean = (X_clean[:20], y_clean[:20])
# val_db = (X_val_clean, y_val_clean)
# tree.fit(X_train_clean, y_train_clean)

accuracy = evaluate(val_db, tree)
print(accuracy)

tree.prune_tree(X_val_noisy, y_val_noisy)

# cross_validation_accuracy = cross_validation(X_noisy, y_noisy)
# print(cross_validation_accuracy)

nested_cross_validation(X_noisy, y_noisy)