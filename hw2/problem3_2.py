import numpy as np
import pdb

np.random.seed(0)

class Dataset:

    def __init__(self, X_tr, ytr, X_te, yte):

        self.num_train = X_tr.shape[0]
        self.indices = np.random.permutation(self.num_train)
        self.X_tr = X_tr[:int(self.num_train*0.9)]
        self.ytr = ytr[:int(self.num_train*0.9)]
        self.X_val = X_tr[int(self.num_train*0.9):]
        self.yval = ytr[int(self.num_train*0.9):]
        self.X_te = X_te
        self.yte = yte

class SimpleLinearNet: 

    def __init__(self, input_size, output_size):

        self.W1 = np.random.randn(input_size, output_size) * 0.01  
        self.b1 = np.zeros((1, output_size))  

    def forward(self, X):
        
        z1 = np.dot(X, self.W1) + self.b1
        y_pred = self.softmax(z1)  
        return y_pred
    
    def softmax(self, x):

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def compute_loss(self, y_pred, y_true, lambda_reg):

        y_pred = np.clip(y_pred, 1e-10, 1.0)
        reg_loss = (lambda_reg / 2) * np.sum(self.W1**2)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        total_loss = loss + reg_loss
        return total_loss
    
    def backward(self, X, y_true, y_pred, learning_rate):

        n = X.shape[0]
        dz1 = (y_pred - y_true) / n
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def accuracy(self, y_pred, y_true):

        predicted_labels = np.argmax(y_pred, axis=1)
        true_labels = y_true.flatten()
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

def train_model(dataset, input_size, output_size, learning_rate, epochs, lambda_reg, batch_size):

    net = SimpleLinearNet(input_size, output_size)  
    yval = np.eye(output_size)[dataset.yval.reshape(-1)]

    num_batches = dataset.X_tr.shape[0] // batch_size

    for epoch in range(epochs):
        for i in range(num_batches):
            X_batch = dataset.X_tr[i * batch_size:(i + 1) * batch_size]
            y_batch = dataset.ytr[i * batch_size:(i + 1) * batch_size]
            y_true = np.eye(output_size)[y_batch.reshape(-1)]
            
            y_pred = net.forward(X_batch)
            loss = net.compute_loss(y_pred, y_true, lambda_reg)
            net.backward(X_batch, y_true, y_pred, learning_rate)    

        if epoch % 10 == 0:
            val_pred = net.forward(dataset.X_val)
            val_loss = net.compute_loss(val_pred, yval, lambda_reg)
            print(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")

    return net, val_loss


def hyperparameter_optimization(dataset, input_size, output_size, num_epochs):

    best_val_loss = float('inf')
    best_params = {}
    best_net = None

    learning_rates = [1e-2, 1e-3]
    regularization_strengths = [0.01, 0.001]
    batch_sizes = [64, 128]

    for lr in learning_rates:
        for reg_strength in regularization_strengths:
            for batch_size in batch_sizes:
                print(f"Training with LR: {lr}, Regularization: {reg_strength}, Batch Size: {batch_size}")

                net, val_loss = train_model(
                    dataset, input_size, output_size, lr, num_epochs, reg_strength, batch_size
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {
                        'learning_rate': lr,
                        'reg_strength': reg_strength,
                        'batch_size': batch_size
                    }
                    best_net = net

    print("Best hyperparameters:")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Regularization Strength: {best_params['reg_strength']}")
    print(f"Batch Size: {best_params['batch_size']}")

    return best_net, best_params


if __name__ == "__main__":

    X_tr = np.reshape(np.load("hw2/problem3/fashion_mnist_train_images.npy"), (-1, 28*28))
    ytr = np.load("hw2/problem3/fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("hw2/problem3/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("hw2/problem3/fashion_mnist_test_labels.npy")

    X_tr = X_tr / 255.0
    X_te = X_te / 255.0

    dataset = Dataset(X_tr, ytr, X_te, yte)

    input_size = 28*28
    output_size = 10
    num_epochs = 100

    best_net, best_params = hyperparameter_optimization(dataset, input_size, output_size, num_epochs)

    test_pred = best_net.forward(dataset.X_te)
    y_test_true = np.eye(output_size)[dataset.yte.reshape(-1)]
    test_loss = best_net.compute_loss(test_pred, y_test_true, lambda_reg=best_params['reg_strength'])
    test_accuracy = best_net.accuracy(test_pred, dataset.yte)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
