import numpy as np
import pdb

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A, B) - C

def problem_1c (A, B, C):
    return A*B + C.T

def problem_1d (x, y):
    return x.T*y

def problem_1e (A, i):
    return np.sum(A[i, 0::2])

def problem_1f (A, c, d):
    return np.mean(A[(A >= c) & (A <= d)])

def problem_1g (A, k):
    eigval, eigvec = np.linalg.eig(A)
    idx = np.argsort(np.abs(eigval))[-k:]
    return eigvec[:, idx]

def problem_1h (x, k, m, s):
    n = len(x)
    z = np.ones(n)
    mean = x.T + m*z
    cov = s*np.eye(n)
    mat = np.random.multivariate_normal(mean, cov, k)
    return mat

def problem_1i (A):
    n = A.shape[0]
    perm = np.random.permutation(n)
    A = A[:, perm]
    return A

def problem_1j (x):
    mean = np.mean(x)
    std = np.std(x)
    y = (x - mean)/std
    return y

def problem_1k (x, k):
    x = x.reshape(-1, 1)
    return np.repeat(x, k, axis=1)

def problem_1l (X, Y):
    X_new = X[:, :, np.newaxis]
    Y_new = Y[:, np.newaxis, :]
    sq_diff = (X_new - Y_new)**2   
    sum_sq_diff = np.sum(sq_diff, axis=0)
    pdb.set_trace()
    return np.sqrt(sum_sq_diff)

# if __name__ == '__main__':
#     print(problem_1h(np.array([1, 2, 3]), 2, 1, 1))
#     print(problem_1i(np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])))
#     print(problem_1j(np.array([1, 2, 3])))
#     print(problem_1k(np.array([1, 2, 3]), 5))
#     print(problem_1l(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [3, 4]])))



    
