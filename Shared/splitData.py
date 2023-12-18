# import dataset
from sklearn.datasets import load_breast_cancer, load_wine
# import pandas and numpy
import pandas as pd
import numpy as np
from math import floor, ceil

class SplitData:
    def __init__(self, X:np.array or pd.DataFrame, y:np.array or pd.DataFrame, test_size:float=0.2, val_size : float = 0, debug: bool = False):
        self.debug = debug
        self.X = X
        # if dataframes make them numpy arrays
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.to_numpy()
        if self.debug: print("X: ", self.X.shape)
        self.y = y
        # if dataframes make them numpy arrays
        if isinstance(self.y, pd.DataFrame):
            self.y = self.y.to_numpy()
        if self.debug: print("y: ", self.y.shape)
        self.test_size = test_size
        if self.debug: print("test_size: ", self.test_size)
        self.val_size = val_size
        if self.debug: print("val_size: ", self.val_size)
        # split data
        self.split()
        if self.debug: print("X_train: ", self.X_train.shape)
        if self.debug: print("y_train: ", self.y_train.shape)
        if self.val_size != 0:
            if self.debug: print("X_val: ", self.X_val.shape)
            if self.debug: print("y_val: ", self.y_val.shape)
        if self.debug: print("X_test: ", self.X_test.shape)
        if self.debug: print("y_test: ", self.y_test.shape)


    def split(self):
        if self.test_size < 0 or self.test_size > 1:
                raise ValueError("Split must be between 0 and 1")
        elif self.val_size < 0 or self.val_size > 1:
                raise ValueError("Split must be between 0 and 1")
        elif sum([self.test_size, self.val_size]) > 1:
                raise ValueError("Sum of splits must be less than 1")
        else:
            permutation = np.random.permutation(self.X.shape[0])
            test = (0,floor(len(permutation)* self.test_size))
            val= ceil(len(permutation)* self.test_size), ceil(len(permutation)* self.test_size) +floor(len(permutation)* self.val_size)
            train = ceil(len(permutation)* self.val_size)+ceil(len(permutation)* self.test_size),len(permutation)

            permutation_test = permutation[test[0]:test[1]+1]
            self.X_test = self.X[permutation_test]
            self.y_test = self.y[permutation_test]

            if self.val_size != 0:
                permutation_val = permutation[val[0]:val[1]+1]
                self.X_val = self.X[permutation_val]
                self.y_val = self.y[permutation_val]
                
            permutation_train = permutation[train[0]:train[1]]
            self.X_train = self.X[permutation_train]
            self.y_train = self.y[permutation_train]
   
    def get_split_data(self):
        if self.val_size != 0:
            return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val
        return self.X_train, self.y_train, self.X_test, self.y_test

def main():
    data_wine = load_wine()
    X = data_wine.data
    y = data_wine.target


    data = SplitData(X, y, debug=True)
    data.get_split_data()

if __name__ == "__main__":
    main()