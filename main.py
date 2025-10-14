import numpy as np


def load_data(filepath):
    X = []
    y = []
    data = np.loadtxt(filepath)
    for row in data:
        X.append(list(map(float,row[0:-1])))
        y.append(int(row[-1]))

    return X, y

    
X_train, y_train = load_data("wifi_db/noisy_dataset.txt")
X_test, y_test = load_data("wifi_db/clean_dataset.txt")
