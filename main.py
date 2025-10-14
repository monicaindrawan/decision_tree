import numpy as np

def load_data(file):
    data = np.loadtxt(file)

    x = data[:, :-1]
    y = data[:, -1].astype(int)

    return x, y

    

X_train, y_train = load_data("wifi_db/noisy_dataset.txt")
X_test, y_test = load_data("wifi_db/clean_dataset.txt")