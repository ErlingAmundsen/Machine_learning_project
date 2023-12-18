import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# set working directory to current directory
if sys.platform == "darwin" or sys.platform == "linux":
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    os.chdir(path)
else:
    path = os.path.realpath(__file__).rsplit("/", 1)[0] #point to a file
    dir = os.path.dirname(path) #point to a directory
    os.chdir(dir)
  


# Based on Piotr Skalski's implementation: 
# https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795 (25/11/2022)
class NeuralNet:
    """Feed forward neural network with backpropagation"""
    def __init__(self):
        self.architecture = []
        self.memory = {}
        self.parameters = {}
        self.gradients = {}
        self.learning_rate = 0.1
        self.l2_reg = 0


    def add_convolutional_layer(self):
        pass


    def add_dense_layer(self, input_size, output_size, activation="ReLu"):
        """Adds a dense layer to the neural network

        Args:
            input_size (int): size of the input to the layer
            output_size (int): number of nodes in the layer
            activation (str, optional): Activation function to apply to the 
            given layer to add non-linearity. Defaults to "ReLu".
        """
        self.architecture.append({
            "input_size": input_size,
            "output_size": output_size,
            "activation": activation
        })

    
    def initialize_parameters(self, seed = 42):
        """Initializes the parameters (weights and biases) of each of the layers.

        Args:
            seed (int, optional): A seed to be set to perserve reproducebility. 
            Defaults to 42.
        """
        np.random.seed(seed)
        for idx, layer in enumerate(self.architecture, 1):
            layer_idx = str(idx)
            self.parameters["W" + layer_idx] = np.random.randn(layer["input_size"], layer["output_size"]) * np.sqrt(2 / layer["input_size"])
            self.parameters["b" + layer_idx] = np.zeros(layer["output_size"])


    def relu(self, Z):
        """Implementation of the rectified linear unit

        Args:
            Z (np.ndarray): an array where the ReLU function whould be 
            applied element-wise

        Returns:
            np.ndarray: Returns the array where any negative elements are 0 else 
            they are preserved
        """
        return np.maximum(0, Z)

    
    def relu_backward(self, dA, Z):
        """The backward step including the derivative of ReLU

        Args:
            dA (np.ndarray): Array of dL/dA 
            Z (np.ndarray): Z array memorized from forward prop to calculate
            dA/dZ

        Returns:
            np.ndarray: Array of dL/dZ
        """
        return dA * (Z > 0)


    def sigmoid(self, Z):
        """Implementation of sigmoid activation function.

        Args:
            Z (np.ndarray): An array of linear combinations of weights, inputs
            + biases

        Returns:
            np.ndarray: Array where the sigmoid function has been applied
            element-wise
        """
        # prevent overflow error with large numbers
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))


    def sigmoid_backward(self, dA, Z):
        """The backward step including the derivative of the Sigmoid function

        Args:
            dA (np.ndarray): Array of dL/dA 
            Z (np.ndarray): Z array memorized from forward prop to calculate
            dA/dZ

        Returns:
            np.ndarray: Array of dL/dZ
        """
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    # helper function to avoid code duplication
    def get_activation_func(self, activation: str, backward=False):
        """Helper function that returns the appropriate activation function based
        one the name of the function

        Args:
            activation (str): A string that specifies which activation function should be returned.
            backward (bool, optional): If true return the backward function else return the forwards. Defaults to False.

        Raises:
            Exception: If a specified function is not implemented then a exception
            will be raised.

        Returns:
            function: The appropriate activation function based on input.
        """
        if activation.lower() == "relu":
            return self.relu if not backward else self.relu_backward
        elif activation.lower() == "sigmoid":
            return self.sigmoid if not backward else self.sigmoid_backward
        elif activation.lower() == "softmax":
            return self.softmax if not backward else self.softmax_backward
        else:
            raise Exception("Activation function not supported")


    def binary_crossentropy(self, y, y_hat):
        """Binary crossentropy loss function.

        Args:
            y (np.array): Array of true labels
            y_hat (np.array): Array of probability predictions for the positive class

        Returns:
            float: A number representing the binary cross entropy loss of the prediction.
        """
        y = y.reshape(-1, 1)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


    def cross_entropy_loss(self, y, y_hat):
        """Multi-class cross entropy loss function.

        Args:
            y (np.ndarray): One hot encoded array of the true labels
            y_hat (np.ndarray): Multi dimensional array including the predicted
            probabilities of each of the classes for each of the inputs.

        Returns:
            float: A number representing the cross entropy loss of the prediction.
        """
        # prevent divide by zero encountered in log error
        y_hat[y_hat == 0] = 1e-15
        log_likelyhood = np.log(y_hat)
        return -np.mean(y * log_likelyhood) + self.l2_regularization()

    def l2_regularization(self):
        """Calculates the L2 regularization term

        Returns:
            float: A number representing the L2 regularization term
        """
        l2 = 0
        for idx, layer in enumerate(self.architecture, 1):
            l2 += np.sum(np.square(self.parameters["W" + str(idx)]))
        return self.l2_reg * l2

    def get_accuracy_score(self, y, y_hat):
        """Accuracy score from true labels and predictions

        Args:
            y (np.array): Array of class labels
            y_hat (np.array): Array of predicted class labels

        Returns:
            float: A number representing the probability of classifying a input correctly 
        """
        return np.sum(y == y_hat) / len(y)


    def one_hot(self, y):
        """Takes array with index values of shape (*) and returns an array of shape (*, self.num_classes) that have
        zeros everywhere except where the index of last dimension matches the corresponding value of the input array, 
        in which case it will be 1.

        Args:
            y (np.array): 1D array of index values

        Returns:
            np.ndarray: 2D array with 1 values at the index of the second dimension indicated by 
            the input, and 0 everywhere else.
        """
        one_hot = np.zeros((y.shape[0], self.num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


    def predict_proba(self, X):
        """Get prediction for the probability of each class label.

        Args:
            X (np.ndarray): Array of input vectors.

        Returns:
            np.ndarray: Array of vectors of probabilities.
        """
        y_hat = self.forward_propagation(X)
        return y_hat


    def softmax(self, Z):
        """Softmax function that converts a vector of real numbers to their relative probabilities.

        Args:
            Z (np.ndarray): Array of vectors of outputs from model.

        Returns:
            np.ndarray: Array of vectors where the softmax function has been aplied to each vector.
        """
        Z_rel = Z - np.max(Z, axis=1, keepdims=True)
        res = np.exp(Z_rel) / np.sum(np.exp(Z_rel), axis=1, keepdims=True)
        return res

    def softmax_backward(self, dA, Z):
        return dA - self.one_hot(self.y)

    def predict_from_proba(self, proba):
        """Preicts the class label from array of vectors with probabilities

        Args:
            proba (np.ndarray): Array of probability vectors

        Returns:
            np.array: Array of class predictions
        """
        if proba.shape[1] == 1:
            return ((proba > 0.5) * 1).flatten()
        else:
            return np.argmax(proba, axis=1)


    def predict(self, X):
        """Predict class labels for an array of inputs.

        Args:
            X (np.ndarray): Array of input vectors.

        Returns:
            np.array: Array of class label prediction based on highest predicted probability.
        """
        preds = self.predict_proba(X)
        return self.predict_from_proba(preds)


    def single_layer_forward_step(self, prev_A, curr_w, curr_b, idx, activation):
        """Single step through one layer of forward propagation.

        Args:
            prev_A (np.ndarray): Array of vectors of activation from previous layer. Input array for first layer.
            curr_w (np.ndarray): Array of weights for the current layer
            curr_b (np.array): Vector of biases for the current layer
            idx (int): Index of current layer
            activation (str): Name of the activation function to be used

        Returns:
            np.ndarray: Z is the array of linear combinations of weights and biases from the input and A is the array
            of values given by applying the activation function.
        """
        Z = prev_A @ curr_w + curr_b

        activation_func = self.get_activation_func(activation)
        A = activation_func(Z)
        return A, Z

    
    def forward_propagation(self, X):
        """Full forward propagation through all the layers.

        Args:
            X (np.ndarray): Array of input vectors.

        Returns:
            np.nparray: Array of vectors where each vector is the set of predicted 
            probabilities for each of the possible classes
        """
        curr_A = X
        for idx, layer in enumerate(self.architecture, 1):
            layer_idx = str(idx)
            prev_A = curr_A
            curr_w = self.parameters["W" + layer_idx]
            curr_b = self.parameters["b" + layer_idx]
            curr_A, curr_Z = self.single_layer_forward_step(prev_A, curr_w, curr_b, idx, layer["activation"])
            self.memory["A" + str(idx - 1)] = prev_A
            self.memory["Z" + layer_idx] = curr_Z
        
        return self.softmax(curr_A)
    

    def single_layer_backward_step(self, curr_dA, layer_idx, activation):
        """Single step backwards through layers for backpropagation.

        Args:
            curr_dA (np.ndarray): Array of the derivatives dL/dA of the current layer.
            layer_idx (int): index of current layer in architecture.
            activation (str): The activation function that was applied to this layes output.

        Returns:
            prev_dA (np.ndarray): Array of the derivatives dL/dA of the previous layer.
            curr_dW (np.ndarray): Array of the derivatives dL/dW for the current layer.
            curr_db (np.ndarray): Array of the derivatives dL/db for the current layer.
        """
        backward_activation_func = self.get_activation_func(activation, backward=True)

        prev_A = self.memory["A" + str(int(layer_idx) - 1)]
        curr_Z = self.memory["Z" + layer_idx]
        curr_W = self.parameters["W" + layer_idx]

        # Derivatives based on the lecture slides and 
        # Mika Senghaas's implementation from exercise 19
        curr_dZ = backward_activation_func(curr_dA, curr_Z)
        curr_dW = prev_A.T @ curr_dZ
        curr_db = curr_dZ.sum(axis=0).reshape(1, -1)
        prev_dA = curr_dZ @ curr_W.T

        return prev_dA, curr_dW, curr_db


    def backward_propagation(self, y, y_hat):
        """Full back propagation cycle through all layers where the calculated 
        gradients are saved to the gradients dictionary.

        Args:
            y (np.ndarray): Array of true labels
            y_hat (np.ndarray): Array of predicted probabilities for each class
        """
        prev_dA = y_hat - self.one_hot(y)

        for idx, layer in reversed(list(enumerate(self.architecture, 1))):
            curr_dA = prev_dA

            prev_dA, curr_dW, curr_db = self.single_layer_backward_step(
                curr_dA = curr_dA, 
                layer_idx = str(idx), 
                activation = layer['activation'])

            self.gradients["dW" + str(idx)] = curr_dW
            # print(curr_db)
            self.gradients["db" + str(idx)] = curr_db
        

    def update_parameters(self):
        """Function to update weights and biases based on the gradients saved in self.gradients.
        """
        for idx, layer in enumerate(self.architecture, 1):
            self.parameters["W" + str(idx)] -= self.learning_rate * self.gradients["dW" + str(idx)]
            self.parameters["b" + str(idx)] = self.parameters["b" + str(idx)] - self.learning_rate * self.gradients["db" + str(idx)]
    

    def fit(self, X, y, epochs=10, learning_rate=0.1, min_improve_epochs=None, minibatch_size=None, val_set=None, lr_scheduler=None, L2_reg=0):
        """Fitting the model on a set of inputs and targets using gradient decend. 

        Args:
            X (np.ndarray): Array of vectors for each training example
            y (np.array): Target vector
            epochs (int, optional): Number of iterations to look at the full Dataset. Defaults to 10.
            learning_rate (float, optional): The step size used for updating parameters during gradient decend. Defaults to 0.1.
            min_improve_epochs (int, optional): The maximum amount of epochs to go through without seeing an improvement. Defaults to 20.
            verbose (bool, optional): Not implemented yet. Defaults to False.
        """
        self.y = y
        self.X = X
        self.learning_rate = learning_rate
        self.num_classes = len(np.unique(y))
        self.initialize_parameters()
        self.loss_hist = []
        self.accuracy_hist = []
        self.loss_hist_val = []
        self.accuracy_hist_val = []
        lr_decay = learning_rate / epochs
        self.l2_reg = L2_reg

        min_loss = np.inf
        for epoch in range(epochs):
            if minibatch_size:
                self.learning_rate *= (1 / (1 + lr_decay * epoch))
                epoch_idx = random.sample(range(X.shape[0]), k = min(32, X.shape[0]))
                self.X = X[epoch_idx]
                self.y = y[epoch_idx]
            else:
                self.X = X
                self.y = y
            y_hat = self.forward_propagation(self.X)
            loss = self.cross_entropy_loss(self.one_hot(self.y), y_hat)
            if min_improve_epochs:
                if loss < min_loss:
                    min_loss = loss
                    best_epoch = epoch
                    best_params = self.parameters
            self.loss_hist.append(loss)
            preds = self.predict_from_proba(y_hat)
            accuracy = self.get_accuracy_score(self.y, preds)
            self.accuracy_hist.append(accuracy)
            self.backward_propagation(self.y, y_hat)
            self.update_parameters()
            if lr_scheduler:
                self.learning_rate = lr_scheduler(self.learning_rate, epoch)

            if val_set:
                val_y_hat = self.forward_propagation(val_set[0])
                val_preds = self.predict_from_proba(val_y_hat)
                val_accuracy = self.get_accuracy_score(val_set[1], val_preds)
                val_loss = self.cross_entropy_loss(self.one_hot(val_set[1]), val_y_hat)
                self.loss_hist_val.append(val_loss)
                self.accuracy_hist_val.append(val_accuracy)


            if val_set:
                _loss = val_loss
            else:
                _loss = loss

            if _loss < min_loss:
                min_loss = _loss
                best_epoch = epoch
                best_params = self.parameters


            # Printing the progress of training
            if val_set:
                print(f"Epoch: {epoch} \tTrain loss: {loss:.5f} \tTraining accuracy: {accuracy:.3%} \tValidation loss: {val_loss:.5f} \tValidation accuracy: {val_accuracy:.3%}", end = "\r")
            else:
                print(f"Epoch: {epoch} \tLoss: {loss:.5f} \tAccuracy: {accuracy:.3%}", end = "\r")

            if epoch % 100 == 0:
                if val_set:
                    print(f"Epoch: {epoch} \tTrain loss: {loss:.5f} \tTraining accuracy: {accuracy:.3%} \tValidation loss: {val_loss:.5f} \tValidation accuracy: {val_accuracy:.3%}")
                else:
                    print(f"Epoch: {epoch} \tLoss: {loss:.5f} \tAccuracy: {accuracy:.3%}")
                    

            if min_improve_epochs:
                if epoch - best_epoch > min_improve_epochs:
                    print()
                    print(f"Early stopping, due to no improvement in loss over {min_improve_epochs} epochs \nBest epcoh: {best_epoch} \tBest Loss: {min_loss:.5f}")
                    break
        self.parameters = best_params
        print()
        
        # print(self.loss_hist)

def create_lr_schedule(start_lr, max_lr, min_lr, total_epochs):
    def lr_scheduler(lr, epoch):
        max_lr_epoch = (total_epochs // 5) * 2
        if epoch % 10 == 0:
            print()
            print(f"Epoch: {epoch} \tLearning rate: {lr:.6f}")
        # linearly scale from start_lr to max_lr during first 40% of epochs
        if epoch < max_lr_epoch:
            return lr + ((max_lr - start_lr) / max_lr_epoch)

        # linearly scale from max_lr to start_lr during next 40% of epochs
        elif epoch < (total_epochs // 5) * 4:
            return lr - ((max_lr - start_lr) / max_lr_epoch)

        # linearly scale from start_lr to min_lr during last 20% of epochs
        else:
            return lr - (start_lr - min_lr) / total_epochs
    
    return lr_scheduler


def train_model(X_train, y_train, epochs = 800, lr=0.0001, min_improve_epochs=None, minibatch_size=None, val_set=None, lr_scheduler=None, L2_reg=0):
    X_dev, y_dev = val_set if val_set else (None, None)

    model = NeuralNet()

    # add layers
    model.add_dense_layer(X_train.shape[1], 50, activation="ReLu") # input -> first layer
    model.add_dense_layer(50, 25, activation="ReLu") # first layer -> second layer
    model.add_dense_layer(25, 10, activation="ReLu") # second layer -> third layer
    model.add_dense_layer(10, 5, activation="Sigmoid") # third layer -> output layer    

    # initialize lr scheduler
    if lr_scheduler:
        lr_scheduler = create_lr_schedule(start_lr=lr, max_lr=lr*10, min_lr=lr/2, total_epochs=epochs)

    model.fit(X_train, y_train, epochs=epochs, learning_rate=lr, min_improve_epochs=min_improve_epochs, minibatch_size=minibatch_size, val_set=val_set, lr_scheduler=lr_scheduler, L2_reg=L2_reg)

    print("\nTrain")
    preds = model.predict(X_train)
    loss = model.cross_entropy_loss(model.one_hot(y_train), model.forward_propagation(X_train))
    acc = model.get_accuracy_score(y_train, preds)
    print(f"Train loss: {loss:.5f}")
    print(f"Train accuracy: {acc:.3%}")
    AUC_roc = roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr")
    print(f"Train AUC ROC: {AUC_roc}")

    if val_set:
        print("\nDev")
        preds = model.predict(X_dev)
        loss = model.cross_entropy_loss(model.one_hot(y_dev), model.forward_propagation(X_dev))
        acc = model.get_accuracy_score(y_dev, preds)
        print(f"Dev loss: {loss:.5f}")
        print(f"Dev accuracy: {acc:.3%}")
        AUC_roc = roc_auc_score(y_dev, model.predict_proba(X_dev), multi_class="ovr")
        print(f"Dev AUC ROC: {AUC_roc}")

    fig = plt.figure(figsize=(10, 5))
    for i, metric in enumerate([model.loss_hist, model.accuracy_hist]):
        plt.subplot(1, 2, i+1)
        plt.plot(range(len(metric)), metric)
        plt.title(f"Loss" if i == 0 else f"Accuracy")
    plt.show()

    return model


def main():
    # load fasion mnist dataset
    train, test = np.load("../Data/fashion_train.npy"), np.load("../Data/fashion_test.npy")

    # split data
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    # split training data into training and dev set
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    # PCA
    run_pca = True
    if run_pca:
        pca = PCA()
        X_train = pca.fit_transform(X_train)
        X_dev = pca.transform(X_dev)
        X_test = pca.transform(X_test)

    # train model and output results
    train_model(X_train, y_train, epochs=1000, lr=0.0001, min_improve_epochs=50, val_set=(X_dev, y_dev), L2_reg=0.0001)

if __name__ == '__main__':
    main()