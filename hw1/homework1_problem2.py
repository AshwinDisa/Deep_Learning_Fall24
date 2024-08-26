import numpy as np
import pdb
import cv2

np.random.seed(0)

class Dataset:

    def __init__(self, X_tr, ytr, X_te, yte):

        # Shuffle the data & split into 90% training and 10% validation
        self.num_train = X_tr.shape[0]
        self.indices = np.random.permutation(self.num_train)
        self.X_tr = X_tr[:int(self.num_train*0.9)]
        self.ytr = ytr[:int(self.num_train*0.9)]
        self.X_val = X_tr[int(self.num_train*0.9):]
        self.yval = ytr[int(self.num_train*0.9):]
        self.X_te = X_te
        self.yte = yte


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size):

        # initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.mazimum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2
    
    def compute_loss(self, y_pred, y_true):

        return np.mean((y_pred - y_true)**2)
    
    def backward(self, X, y_true, y_pred, learning_rate):





# def train_age_regressor():

if __name__ == "__main__":

    # import data
    X_tr = np.reshape(np.load("hw1/data/age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("hw1/data/age_regression_ytr.npy")
    X_te = np.reshape(np.load("hw1/data/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("hw1/data/age_regression_yte.npy")

    # create dataset
    dataset = Dataset(X_tr, ytr, X_te, yte)

    # parameters
    input_size = 48*48
    hidden_size = 64
    output_size = 1
    learning_rate = 1e-3
    num_epochs = 10

    # train the network
    net = TwoLayerNet(input_size, hidden_size, output_size)


