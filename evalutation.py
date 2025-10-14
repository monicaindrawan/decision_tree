from main import *
import numpy as np

def evaluate(x_test, y_test, trained_tree=None):
    #x_test, y_test = test_db
    test_data = np.array_split(x_test, 10)
    
    for i in range(len(test_data)):
        test = test_data[i]
        train = []
        train.append(test_data[:i])
        train.append(test_data[i+1:])
    



    

evaluate(X_test, y_test)
