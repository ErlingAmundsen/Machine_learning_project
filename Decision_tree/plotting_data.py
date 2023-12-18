import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import numpy as np

import pandas as pd

## creating dataset in the form given by load_wine()

# data_wine = load_wine()
# X = data_wine.data
# y = data_wine.target
# feature_names = data_wine.feature_names

## give even number

# def data_creator(data_rows = 100, features = 3):
#     X = np.zeros((data_rows, features+1))
#     for i in range(features):
#         x_class_1 = list(np.random.random(int(data_rows/2)))
#         x_class_2 = list(np.random.random(int(data_rows/2))+i)
#         X[0:int(data_rows/2),i] = x_class_1
#         X[int(data_rows/2):,i] = x_class_2
    
#     for i in range(data_rows):
#         if i < int(data_rows/2):
#             X[i,features] = 0
#         else:
#             X[i,features] = 1
    
    
#     return X[:,:features],X[:,features] 

def simple_2_class():
    Dataset_2_classes = np.zeros((100,3))
    
    # feature 1
    Dataset_2_classes[:50,0] = np.random.random(50)
    Dataset_2_classes[50:,0] = np.random.random(50)+1.1
    
    # feature 2
    Dataset_2_classes[:50,1] = np.random.random(50)
    Dataset_2_classes[50:,1] = np.random.random(50)
    
    # class
    Dataset_2_classes[:50,2] = 0
    Dataset_2_classes[50:,2] = 1
    return Dataset_2_classes[:,:2],Dataset_2_classes[:,2]

def medium_2_class():
    Dataset_2_classes = np.zeros((100,3))
    
    # feature 1
    Dataset_2_classes[:50,0] = np.random.random(50)
    Dataset_2_classes[50:,0] = np.random.random(50)
    
    # feature 2
    Dataset_2_classes[:50,1] = np.random.random(50)
    Dataset_2_classes[50:,1] = np.random.random(50)+.4
    
    # class
    Dataset_2_classes[:50,2] = 0
    Dataset_2_classes[50:,2] = 1
    return Dataset_2_classes[:,:2],Dataset_2_classes[:,2]

def simple_3_class():
    Dataset_2_classes = np.zeros((150,3))
    
    # feature 1
    Dataset_2_classes[:50,0] = np.random.random(50)
    Dataset_2_classes[50:100,0] = np.random.random(50)+1.1
    Dataset_2_classes[100:150,0] = np.random.random(50)+2.2
    
    # feature 2
    Dataset_2_classes[:50,1] = np.random.random(50)
    Dataset_2_classes[50:100,1] = np.random.random(50) +1.1
    Dataset_2_classes[100:150,1] = np.random.random(50)
    
    # class
    Dataset_2_classes[:50,2] = 0
    Dataset_2_classes[50:100,2] = 1
    Dataset_2_classes[100:150,2] = 2
    
    return Dataset_2_classes[:,:2],Dataset_2_classes[:,2]

def medium_3_class():
    Dataset_2_classes = np.zeros((150,3))
    
    # feature 1
    Dataset_2_classes[:50,0] = np.random.random(50)
    Dataset_2_classes[50:100,0] = np.random.random(50)+0.5
    Dataset_2_classes[100:150,0] = np.random.random(50)+1.1
    
    # feature 2
    Dataset_2_classes[:50,1] = np.random.random(50)
    Dataset_2_classes[50:100,1] = np.random.random(50) +1.1
    Dataset_2_classes[100:150,1] = np.random.random(50)
    
    # class
    Dataset_2_classes[:50,2] = 0
    Dataset_2_classes[50:100,2] = 1
    Dataset_2_classes[100:150,2] = 2
    
    return Dataset_2_classes[:,:2],Dataset_2_classes[:,2]





def main():
    X, Y = medium_3_class()
    plt.scatter(X[:,0],X[:,1], c = Y)
    plt.show()
    return

if __name__ == '__main__':
    main()
    
        
        