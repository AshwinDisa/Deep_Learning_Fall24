import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 64 ]
NUM_OUTPUT = 10

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
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

def fCE (X, Y, weights):
    Ws, bs = unpack(weights)
    # ...
    return ce
   
def gradCE (X, Y, weights):
    Ws, bs = unpack(weights)
    # ...
    return allGradientsAsVector

# Creates an image representing the first layer of weights (W0).
def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()
 
def train (trainX, trainY, weights, testX, testY, lr = 5e-2):
    # TODO: implement me
    pass

def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight using a variant of the Kaiming He Uniform technique.
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i+1], NUM_HIDDEN[i]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs

if __name__ == "__main__":
    # Load training data.
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).

    Ws, bs = initWeightsAndBiases()

    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).
    print(scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_), \
                                    lambda weights_: gradCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_), \
                                    weights))
    #print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_), 1e-6))

    weights = train(trainX, trainY, weights, testX, testY, 0.01)
    show_W0(weights)
