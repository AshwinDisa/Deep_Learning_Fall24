import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pdb

NUM_HIDDEN_LAYERS = 1
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [64]
NUM_OUTPUT = 10

def unpack(weights):
    Ws = []
    start = 0
    end = NUM_INPUT * NUM_HIDDEN[0]
    W = weights[start:end].reshape(NUM_HIDDEN[0], NUM_INPUT)
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i] * NUM_HIDDEN[i+1]
        W = weights[start:end].reshape(NUM_HIDDEN[i+1], NUM_HIDDEN[i])
        Ws.append(W)

    start = end
    end = end + NUM_OUTPUT * NUM_HIDDEN[-1]
    W = weights[start:end].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])
    Ws.append(W)

    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def fCE(X, Y, weights):
    Ws, bs = unpack(weights)
    m = X.shape[0]

    A = X
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.dot(W, A.T).T + b
        A = np.maximum(Z, 0)

    ZL = np.dot(Ws[-1], A.T).T + bs[-1]
    A_final = np.exp(ZL) / np.sum(np.exp(ZL), axis=1, keepdims=True) 

    log_likelihood = -np.log(A_final[range(m), Y])
    loss = np.sum(log_likelihood) / m
    return loss

def gradCE(X, Y, weights):
    Ws, bs = unpack(weights)
    m = X.shape[0]

    A = X
    cache_A = [A]
    cache_Z = []

    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.dot(A, W.T) + b
        A = np.maximum(Z, 0)
        cache_A.append(A)
        cache_Z.append(Z)

    ZL = np.dot(A, Ws[-1].T) + bs[-1]
    A_final = np.exp(ZL) / np.sum(np.exp(ZL), axis=1, keepdims=True)

    grads_W = []
    grads_b = []

    dZL = A_final
    dZL[range(m), Y] -= 1
    dZL /= m

    dW = np.dot(dZL.T, cache_A[-1])
    db = np.sum(dZL, axis=0)

    grads_W.append(dW)
    grads_b.append(db)

    dA = np.dot(dZL, Ws[-1])

    for i in reversed(range(NUM_HIDDEN_LAYERS)):
        dZ = dA * (cache_Z[i] > 0)
        dW = np.dot(dZ.T, cache_A[i])
        db = np.sum(dZ, axis=0)

        grads_W.append(dW)
        grads_b.append(db)

        dA = np.dot(dZ, Ws[i])

    grads_W.reverse()
    grads_b.reverse()

    all_gradients = np.hstack([g.flatten() for g in grads_W] + [g.flatten() for g in grads_b])
    return all_gradients

def train(trainX, trainY, weights, testX, testY, lr=5e-2, epochs=100, batch_size=128):
    m = trainX.shape[0]

    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = trainX[i:i+batch_size]
            Y_batch = trainY[i:i+batch_size]

            gradients = gradCE(X_batch, Y_batch, weights)
            weights -= lr * gradients

        test_loss = fCE(testX, testY, weights)
        predictions = predict(testX, weights)
        accuracy = np.mean(predictions == testY)

        print(f'Epoch {epoch+1}/{epochs} - Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    return weights

def predict(X, weights):
    Ws, bs = unpack(weights)
    A = X
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.dot(W, A.T).T + b
        A = np.maximum(Z, 0)

    ZL = np.dot(Ws[-1], A.T).T + bs[-1]
    A_final = np.exp(ZL) / np.sum(np.exp(ZL), axis=1, keepdims=True)
    return np.argmax(A_final, axis=1)

def initWeightsAndBiases():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT)).astype(np.float32) / NUM_INPUT**0.5) - 1. / NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0], dtype=np.float32)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN[i+1], NUM_HIDDEN[i])).astype(np.float32) / NUM_HIDDEN[i]**0.5) - 1. / NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1], dtype=np.float32)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1])).astype(np.float32) / NUM_HIDDEN[-1]**0.5) - 1. / NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT, dtype=np.float32)
    bs.append(b)

    return Ws, bs

def show_W0(weights):

    Ws, _ = unpack(weights)
    W0 = Ws[0]

    num_filters = W0.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < num_filters:
            weight_image = W0[i].reshape(28, 28)
            ax.imshow(weight_image, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    # using turing
    plt.savefig('W0.png')

if __name__ == "__main__":
    trainX = np.load('fashion_mnist_train_images.npy').astype(np.float32)
    trainY = np.load('fashion_mnist_train_labels.npy')
    testX = np.load('fashion_mnist_test_images.npy').astype(np.float32)
    testY = np.load('fashion_mnist_test_labels.npy')

    trainX = trainX / 255.0
    testX = testX / 255.0

    Ws, bs = initWeightsAndBiases()
    weights = np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

    check_grad_result = scipy.optimize.check_grad(
        lambda w_: fCE(trainX[:2], trainY[:2], w_),
        lambda w_: gradCE(trainX[:2], trainY[:2], w_),
        weights,
    )
    print(f'Gradient check discrepancy: {check_grad_result:.6f}')

    weights = train(trainX, trainY, weights, testX, testY)

    show_W0(weights)
