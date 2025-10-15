import numpy as np
from Tree import *

def load_data(file):
    data = np.loadtxt(file)

    x = data[:, :-1]
    y = data[:, -1].astype(int)

    return x, y

    

X_noisy, y_noisy = load_data("wifi_db/noisy_dataset.txt")
X_clean, y_clean = load_data("wifi_db/clean_dataset.txt")


tree = DecisionTree()
tree.fit(X_noisy[20:], y_noisy[20:])
prediction = tree.predict(X_noisy[:20])
print(prediction)