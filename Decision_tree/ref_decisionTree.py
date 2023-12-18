# sklearn decision tree
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../")
sys.path.append("./")


def load_data():
    train = np.load('./Data/fashion_train.npy')
    test = np.load('./Data/fashion_test.npy')
    
    # split data into features and labels where last column is label
    x_data, y_data = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    x_data = x_data > 15
    x_test = x_test > 15

    # split training set into train and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    return x_data, y_data, x_train, y_train, x_val, y_val, x_test, y_test


def normalize(Z):
    return Z / np.sum(Z)


def main():
    # initialize the model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # load data
    x_data, y_data, x_train, y_train, x_val, y_val, x_test, y_test = load_data()


    do_pca = False
    if do_pca:
        pca = PCA()
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_data = pca.fit_transform(x_data)
        x_test = pca.transform(x_test)

    print("train shape")
    print(x_train.shape, y_train.shape)
    uni, cnt = np.unique(y_train, return_counts=True)
    print(f"{uni} {cnt} {normalize(cnt)} \n")

    print("validation shape")
    print(x_val.shape, y_val.shape)
    uni, cnt = np.unique(y_val, return_counts=True)
    print(f"{uni} {cnt} {normalize(cnt)} \n")

    print("test shape")
    print(x_test.shape, y_test.shape)
    print(np.unique(y_test, return_counts=True), "\n")

    # train the model
    model.fit(x_train, y_train)

    # grid search for best hyperparameters
    do_grid_search = False
    if do_grid_search:
        from sklearn.model_selection import GridSearchCV
        param_grid = [
            {'max_leaf_nodes': list(range(81, 85))}
        ]
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=3)
        grid_search.fit(x_train, y_train)

        grid_search.best_params_

    # evaluate the model
    print("train accuracy")
    print(model.score(x_train, y_train), "\n")
    print("validation accuracy")
    print(model.score(x_val, y_val), "\n")

    model_final = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_final.fit(x_data, y_data)

    print("test accuracy")
    print(model_final.score(x_test, y_test), "\n")


if __name__ == '__main__':
    main()