import os
import sys
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# set working directory to current directory
if sys.platform == "darwin" or sys.platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)
  
# import method from sibling module

sys.path.append('../')

from Shared.binaryTree import BinaryTreeNode


class DecisionTree:
    """_summary_
    """
    
    def __init__(self, X, y, feature_names:list = None, debug:bool=False, fullDebug:bool=False, random_cutoffs:bool=True, random_forest:bool=False, impurity_type:str="gini", max_depth:int=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            feature_names (list, optional): _description_. Defaults to None.
            test_size (float, optional): _description_. Defaults to .2.
            val_size (float, optional): _description_. Defaults to 0.1.
            debug (bool, optional): _description_. Defaults to False.
            fullDebug (bool, optional): _description_. Defaults to False.
        """
        if max_depth != None and max_depth < 1:
                raise ValueError("maxdepth must be 1 or more")
            
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.tree = None
        self.debug = debug
        self.classes = np.unique(y)
        self.random_cutoffs = random_cutoffs
        self.random_forest = random_forest
        self.max_depth = max_depth

            
        giniIndex = lambda t, total : np.float_power(t/total,2) # calculate gini index (n: number of true, total: total number of data points)
        giniImpurity = lambda l, total: 1 - sum([giniIndex(l[i], total) for i in range(len(l))])
        newGiniImpurity = lambda l: 1 - np.sum(np.float_power(l/np.sum(l, axis=1).reshape(l.shape[0], 1),2), axis=1)
        entropyIndex = lambda n, total : -((n/total)*np.log2(n/total))
        entropy = lambda l, _: sum([entropyIndex(n, sum(l)) for n in l])

        if impurity_type == "entropy":
            self.impurity = entropy
        else:
            self.newGini = newGiniImpurity
            self.impurity = giniImpurity

        if fullDebug: self.debug = True
        self.fullDebug = fullDebug
                
        if debug: print("X: ", self.X.shape)
        if debug: print("y: ", self.y.shape)
        if debug: print("feature_names: ", self.feature_names)

    
    def calc_impurity(self, data, y):

        # finds classes and counts of each class
        classes, total_each_class = np.unique(y, return_counts = True)
     
        
        if len(total_each_class) == 1:
            return None, None, None
        
        if self.random_forest:
            looking_at_features = np.random.choice(range(data.shape[1]), int(np.floor(np.sqrt(data.shape[1]))), replace=False)
        else:
            looking_at_features = range(data.shape[1])   

        # find best cutoff for each feature
        best_cutoffs = np.array([self.find_best_cutoff(data, y, feature, classes, total_each_class) for feature in tqdm(looking_at_features, leave=False)], dtype=object)

        # find best impurity
        best_impurity = np.min(best_cutoffs[:,0])
        # find index of best impurity
        idx = np.where(best_cutoffs[:,0] == best_impurity)[0][0]
        # find best cutoff and feature
        best_cutoff = best_cutoffs[idx, 1]
        best_feature = best_cutoffs[idx, 2]

        return  best_impurity, best_cutoff, best_feature

    def find_best_cutoff(self, data, y, feature, classes, total_each_class):
        temp = data[data[:, feature].argsort()]
        temp_y = y[data[:, feature].argsort()]

        # create np.array of values to check for cutoffs
        comparison1 = temp[:-1, feature]
        comparison2 = temp[1:, feature]

        # make the two np.arrays to one np.array with multiple np.arrays inside of size 2
        cutoff_comparisons_ = np.concatenate((comparison1.reshape(comparison1.shape[0],1), comparison2.reshape(comparison2.shape[0], 1)), axis=1)
        cutoff_mask = cutoff_comparisons_[:,0]!=cutoff_comparisons_[:,1]

        # if all values are the same, skip this feature
        if cutoff_mask.sum() == 0:
            return 10, None, None
        
        # remove all rows where the two values are the same, to avoid a cuttoff at a datapoint
        cutoff_comparisons = cutoff_comparisons_[cutoff_mask]
        # calculate the cutoffs by taking the average of the two values
        # cutoffs_ = np.average(cutoff_comparisons, axis=1, keepdims=True)

        # create np.array of values to check for classes
        comparison3 = temp_y[:-1]
        comparison4 = temp_y[1:]
        # make the two np.arrays to one np.array with multiple np.arrays inside of size 2
        classes_comparisons_ = np.concatenate((comparison3.reshape(comparison3.shape[0],1), comparison4.reshape(comparison4.shape[0], 1)), axis=1)
        classes_comparisons = classes_comparisons_[cutoff_mask]
        # finds cutoffs where the classes are different
        compared_classes_ = np.equal(classes_comparisons[:,0],classes_comparisons[:,1])
        compared_classes = compared_classes_.reshape(compared_classes_.shape[0], 1)
        cutoff_comparisons_final_ = cutoff_comparisons[np.argwhere(compared_classes != True)]

        if cutoff_comparisons_final_.shape[0] == 0:
            return 10, None, None

        cutoff_comparisons_final = cutoff_comparisons_final_[:,0]


        cutoffs = np.average(cutoff_comparisons_final, axis=1, keepdims=True).reshape(-1)
        # if there are no cutoffs, continue to next feature
        if cutoffs.shape[0] == 0:
            return 10, None, None

        if self.random_cutoffs:
            n_random_cutoffs = 2 * int(np.floor(np.sqrt(temp.shape[0])))
            if len(cutoffs) > n_random_cutoffs:
                cutoffs = np.random.choice(cutoffs, n_random_cutoffs, replace=False)


        # find classes that are less than the cutoff
        under_cutoffs = [temp_y[temp[:, feature] < cutoff] for cutoff in cutoffs]

        # find class and count of each class under the cutoff
        classes_counts = np.array([np.unique(under_cutoff, return_counts = True) for under_cutoff in under_cutoffs], dtype=object)
        under_cutoffs_classes = classes_counts[:,0]
        under_cutoffs_counts = classes_counts[:,1]
        # match shape of under_cutoff_count and total_each_class
        under_cutoffs_counts = np.array([np.array([under_cutoff_count[np.where(under_cutoff_classes == c)[0][0]] if c in under_cutoff_classes else 0 for c in classes]) for under_cutoff_classes, under_cutoff_count in zip(under_cutoffs_classes, under_cutoffs_counts)])

        
        
        # calculate the impurity of each cutoff
        under_wheights = np.sum(under_cutoffs_counts, axis=1)/np.sum(total_each_class)
        over_wheights = np.sum(total_each_class - under_cutoffs_counts, axis=1)/np.sum(total_each_class)

        under = self.newGini(under_cutoffs_counts)
        over_ = total_each_class - under_cutoffs_counts
        over = self.newGini(over_)

        impurity = under_wheights * under + over_wheights * over

        # find the cutoff with the lowest impurity
        temp_best_impurity = np.min(impurity)
        idx = np.where(impurity == temp_best_impurity)[0]
        temp_best_cutoff = cutoffs[idx]

        return temp_best_impurity, temp_best_cutoff, feature


    def create_tree(self, data, y, prev_impurity = None, prev_cutoff = None, prev_feature = None, depth = 1):
            
        if self.debug: print("\nCreating tree...")
        if self.debug: print("data: ", data.shape)
        # print("data: ", data.shape[0])
        impurity, cutoff, feature = self.calc_impurity(data, y)
        if self.debug: print("impurity: ", impurity)
        if self.debug: print("cutoff: ", cutoff)
        if self.debug: print("feature: ", feature)
        
        if impurity == None or prev_impurity <= impurity or (self.max_depth != None and depth == self.max_depth):
            if impurity == None :
                impurity = 0
            return BinaryTreeNode({'class': np.bincount(y).argmax(), 'impurity':impurity, 'depth': depth, 'leaf':True})

        else:

            current_node = BinaryTreeNode({'impurity': impurity, 'cutoff': cutoff, 'feature': feature, 'depth': depth, 'leaf':False, 'count_datapoints': data.shape[0]})
            depth += 1
            current_node.leftChild = self.create_tree(data[data[:,feature] < cutoff], y[data[:,feature] < cutoff], impurity, cutoff, feature, depth)
            current_node.rightChild = self.create_tree(data[data[:,feature] > cutoff], y[data[:,feature] > cutoff], impurity, cutoff, feature, depth)

            return current_node

    def fit(self):
        impurity, cutoff, feature = self.calc_impurity(self.X, self.y)
        root = BinaryTreeNode({'impurity': impurity, 'cutoff': cutoff, 'feature': feature, 'depth': 0, 'leaf':False,  'count_datapoints': self.X.shape[0]})
        
        root.leftChild = self.create_tree(self.X[self.X[:,feature] < cutoff], self.y[self.X[:,feature] < cutoff], impurity, cutoff, feature)
        root.rightChild = self.create_tree(self.X[self.X[:,feature] > cutoff], self.y[self.X[:,feature] > cutoff], impurity, cutoff, feature)
        self.tree = root
        return root
        
    def predict(self, X):
        if self.tree == None:
            raise ValueError("Tree not created yet")
        elif X.shape[1] != self.X.shape[1]:
            raise ValueError("X must have same number of features as training data")
        elif X.shape[0] <= 0:
            raise ValueError("X must have at least one row")
        else:
            predictions = np.zeros((X.shape[0],1))
            for i in range(X.shape[0]):
                root_data = self.tree.data
                root = self.tree
                while root_data['leaf'] == False:
                    cutoff = root_data['cutoff']
                    feature = root_data['feature']
                    if X[i, feature] < cutoff:
                        root_data = root.leftChild.data
                        root = root.leftChild
                    else:
                        root_data = root.rightChild.data
                        root = root.rightChild
                predictions[i] = root_data['class']
        return predictions
    
    def score(self, X, y):
        if self.tree == None:
            raise ValueError("Tree not created yet")
        elif X.shape[1] != self.X.shape[1]:
            raise ValueError("X must have same number of features as training data")
        elif X.shape[0] <= 0:
            raise ValueError("X must have at least one row")
        else:
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        
    def confusion_matrix(self, X, y, labels):
        predictions = self.predict(X).reshape(1,-1)[0]
        
        cm = confusion_matrix(y, predictions)

        cm = cm.astype('float') / cm.sum()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2%', xticklabels=labels, yticklabels=labels, annot_kws={"size":20})
        plt.ylabel('Actual', fontsize=20)
        plt.xlabel('Predicted', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig("cm_dct.png")
         
def main():
    data = np.load('../Data/fashion_train.npy')
    X = data[:,:-1]
    y = data[:,-1]
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    np.random.seed(0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    run_pca = True
    random_cutoffs = True

    if run_pca:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
    
    clf = DecisionTree(X_train, y_train, debug = False,random_cutoffs=random_cutoffs, random_forest=False, max_depth=4)
    start = time.time()
    clf.fit()

    print("PCA: ", run_pca)
    print("Random cutoffs: ", random_cutoffs)
    print("Time taken: ", time.time() - start)

    print(clf.score(X_train, y_train))
    print(clf.score(X_val, y_val))
    
if __name__ == "__main__":
    main()
    