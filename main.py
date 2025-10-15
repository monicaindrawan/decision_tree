import numpy as np
from Tree import DecisionTree
from evaluation import evaluate, cross_validation

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

accuracy = evaluate(val_db, tree)
print(accuracy)

cross_validation_accuracy = cross_validation(X_noisy, y_noisy)
print(cross_validation_accuracy)