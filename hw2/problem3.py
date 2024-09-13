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

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):

        self.z2 = np.dot(X, self.W1) + self.b1
        self.a2 = np.maximum(0, self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.b2
        self.a3 = self.z3
        self.a3 = self.softmax(self.a3)

        return self.a3
    
    def softmax(self, x):

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def compute_loss(self, y_pred, y_true, num_classes, lambda_reg):

        # TODO: regularization
        y_pred = np.clip(y_pred, 1e-10, 1.0)

        reg_loss = (lambda_reg / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        total_loss = loss + reg_loss   

        return loss
    
    def backward(self, X, y_true, y_pred, learning_rate):

        n = X.shape[0]
        # y_true = y_true.reshape(-1, 1)
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
    
    def accuracy(self, y_pred, y_true):
    
        # Convert predictions to class labels
        predicted_labels = np.argmax(y_pred, axis=1)
        true_labels = y_true.flatten()
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

def train_age_regressor(dataset, input_size, hidden_size, output_size, learning_rate, epochs):

    net = TwoLayerNet(input_size, hidden_size, output_size)

    dataset.yval = np.eye(output_size)[dataset.yval.reshape(-1)]

    for epoch in range(epochs):
        
        y_pred = net.forward(dataset.X_tr)
        y_true = np.eye(output_size)[dataset.ytr.reshape(-1)]
        loss = net.compute_loss(y_pred, y_true, output_size, lambda_reg=0.01)
        net.backward(dataset.X_tr, y_true, y_pred, learning_rate)    
        if epoch % 10 == 0:
            val_pred = net.forward(dataset.X_val)
            val_loss = net.compute_loss(val_pred, dataset.yval, output_size, lambda_reg=0.01)
            print(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")

        if epoch > epochs - 10:
            print(f"Training Loss: {loss}, Validation Loss: {val_loss}")

        if epoch == epochs - 1:
            print(f"Training Loss: {loss}, Validation Loss: {val_loss}")

    return net


if __name__ == "__main__":

    # import data
    X_tr = np.reshape(np.load("hw2/problem3/fashion_mnist_train_images.npy"), (-1, 28*28))
    ytr = np.load("hw2/problem3/fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("hw2/problem3/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("hw2/problem3/fashion_mnist_test_labels.npy")

    # create dataset
    dataset = Dataset(X_tr, ytr, X_te, yte)

    # parameters
    input_size = 28*28
    hidden_size = 64
    output_size = 10
    learning_rate = 1e-3
    num_epochs = 500

    # train the network
    net = train_age_regressor(dataset, input_size, hidden_size, output_size, learning_rate, num_epochs)
    test_pred = net.forward(dataset.X_te)
    y_test_true = np.eye(output_size)[dataset.yte.reshape(-1)]
    test_loss = net.compute_loss(test_pred, y_test_true, output_size, lambda_reg=0.01)
    test_accuracy = net.accuracy(test_pred, dataset.yte)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


