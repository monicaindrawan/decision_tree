import numpy as np

def load_data(file):
    data = np.loadtxt(file)

    x = data[:, :-1]
    y = data[:, -1].astype(int)

    return x, y

    

def decision_tree_learning():
    pass

clean_data = load_data("wifi_db/noisy_dataset.txt")
noisy_data = load_data("wifi_db/clean_dataset.txt")

print(clean_data)
print(noisy_data)