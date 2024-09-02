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

        self.z2 = np.dot(X, self.W1) + self.b1
        self.a2 = np.maximum(0, self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.b2
        self.a3 = self.z3
        return self.a3
    
    def compute_loss(self, y_pred, y_true):

        return np.mean((y_pred - y_true)**2)
    
    def backward(self, X, y_true, y_pred, learning_rate):

        n = X.shape[0]
        y_true = y_true.reshape(-1, 1)
        dz3 = (y_pred - y_true) / n
        dW2 = np.dot(self.a2.T, dz3)
        db2 = np.sum(dz3, axis=0, keepdims=True)
        da2 = np.dot(dz3, self.W2.T)
        dz2 = da2 * (self.z2 > 0)
        dW1 = np.dot(X.T, dz2)
        db1 = np.sum(dz2, axis=0, keepdims=True)

        # update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1      

        pdb.set_trace()

def train_age_regressor(dataset, input_size, hidden_size, output_size, learning_rate, epochs):

    net = TwoLayerNet(input_size, hidden_size, output_size)

    for epoch in range(epochs):

        y_pred = net.forward(dataset.X_tr)
        loss = net.compute_loss(y_pred, dataset.ytr)
        net.backward(dataset.X_tr, dataset.ytr, y_pred, learning_rate)    
        if epoch % 100 == 0:
            val_pred = net.forward(dataset.X_val)
            val_loss = net.compute_loss(val_pred, dataset.yval)
            print(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")

        if epoch > epochs - 10:
            print(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")

    return net


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
    learning_rate = 1e-2
    num_epochs = 1000

    # train the network
    net = train_age_regressor(dataset, input_size, hidden_size, output_size, learning_rate, num_epochs)
    test_pred = net.forward(dataset.X_te)
    test_loss = net.compute_loss(test_pred, dataset.yte)

    print(f"Test Loss: {test_loss}")


