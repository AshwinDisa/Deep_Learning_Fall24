
import pdb
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

def problem3a():
    
    # section i
    def func(x):
        if np.isfinite(x) == False:
            return np.nan
        
        if x<-100:
            out = np.nan
        elif x<-0.1:
            out = -x**3
        elif x<3:
            out = -0.03*x - 1/500
        elif x<5:
            out = -(x-3.1)**3 - 23/250
        elif x<100:
            out = 0.5*10.83*(x-6)**2 - 6183/500
        else:
            out = np.nan
                
        return out

    def gradient(x):
        if np.isfinite(x) == False:
            return np.nan
        
        if x<-100:
            out = np.nan
        elif x<-0.1:
            out = -3*x*x
        elif x<3:
            out = -0.03
        elif x<5:
            out = -3*(x-3.1)**2
        elif x<100:
            out = 10.83*(x-6)
        else:
            out = np.nan

        return out
    
    def gradientDescent(x0, lr):
        x = x0
        xs = []
        ys = []
        grads = []
        for index in range(100):
            y = func(x)
            grad = gradient(x)

            if np.isnan(x) or np.isnan(grad):
                break

            xs.append(x)
            ys.append(y)
            grads.append(grad)

            x -= lr*grad

        return np.array(xs), np.array(ys), np.array(grads)

    def plotSettings(plt, title, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.minorticks_on()
        plt.legend()
        plt.show()

    lrList = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
    x0 = 10
    for lr in lrList:
        x, y, g = gradientDescent(x0, lr)
        plt.plot(x, label="LR: " + str(lr))
        # plt.plot(g, label="LR: " + str(lr))
        # plt.plot(y, label="LR: " + str(lr))

    plotSettings(plt, "X over iteration", "Iteration", "X")
    # plotSettings(plt, "Grad over iteration", "Iteration", "Gradient")
    # plotSettings(plt, "func over iteration", "Iteration", "f")
    # pdb.set_trace()

def problem3b():
    def func(x, param):
        a1, a2, c1, c2 = param
        out = a1*(x[0]-c1)**2 + a2*(x[1]-c2)**2
                
        return out

    def gradient(x, param):
        a1, a2, c1, c2 = param
        g1 = 2*a1*(x[0] - c1)
        g2 = 2*a2*(x[1] - c2)

        return np.array([g1, g2])
    
    def gradientDescent(x0, lr):
        x = x0
        xs = []
        ys = []
        grads = []
        a1 = 100
        a2 = 1
        c1 = 0.4
        c2 = 4
        param = [a1, a2, c1, c2]

        for index in range(200):
            y = func(x, param)
            grad = gradient(x, param)

            # if np.isnan(x).any() or np.isnan(grad).any():
            #     break

            xs.append(x.copy())
            ys.append(y.copy())
            grads.append(grad.copy())
            
            x = x - lr*grad
            

        return np.array(xs), np.array(ys), np.array(grads)
    
    def plotSettings(plt, title, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.minorticks_on()
        # plt.legend()
        plt.show()
    
    x0 = np.array([1, -3.])
    x, y, g = gradientDescent(x0, 0.0095)
    # pdb.set_trace()
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot(x[:, 0], x[:, 1], color='red')

    plotSettings(plt, "x2 vs x1 Scatter", "x1", "y2")

def problem3c():
    
    # section i
    def func(x):
        if np.isfinite(x) == False:
            return np.nan
        
        out = 2/3*np.power(np.abs(x), 3/2)
                
        return out

    def gradient(x):
        if np.isfinite(x) == False:
            return np.nan
        
        if x>0:
            out = np.sqrt(x)
        elif x<0:
            out = -np.sqrt(-x)
        else:
            out = 0

        return out
    
    def gradientDescent(x0, lr):
        x = x0
        xs = []
        ys = []
        grads = []
        for index in range(20):
            y = func(x)
            grad = gradient(x)

            if np.isnan(x) or np.isnan(grad):
                break

            xs.append(x)
            ys.append(y)
            grads.append(grad)

            x -= lr*grad

        return np.array(xs), np.array(ys), np.array(grads)

    def plotSettings(plt, title, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.minorticks_on()
        plt.legend()
        plt.show()

    lrList = [1e-3, 1e-2, 1e-1]
    x0 = -0.01
    for lr in lrList:
        x, y, g = gradientDescent(x0, lr)
        # plt.plot(x, label="LR: " + str(lr))
        # plt.plot(g, label="LR: " + str(lr))
        # plt.plot(y, label="LR: " + str(lr))
        plt.scatter(x, y, label="LR: " + str(lr))
        for i in range(len(x)):
            plt.text(x[i], y[i], f'{i}', fontsize=9, ha='right')

    plotSettings(plt, f"Y vs X with x0={x0}", "X", "Y")


if __name__ == "__main__":
    # problem3a()
    problem3b()
    # problem3c()