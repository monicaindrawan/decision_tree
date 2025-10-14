import numpy as np

def load_data(file):
    data = np.loadtxt(file)
    return data
    

def decision_tree_learning():
    pass

clean_data = load_data("wifi_db/noisy_dataset.txt")
noisy_data = load_data("wifi_db/clean_dataset.txt")

x = clean_data[:,:-1]
y = clean_data[:,-1]
print(x[0,0])
print(type(x[0,0]))