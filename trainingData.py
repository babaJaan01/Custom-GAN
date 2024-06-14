import numpy as np
import matplotlib.pyplot as plt

def getY(x):
    return x * x + 10 # just the function y = x^2 + 10

def sampleData(n = 10000, scale = 100): # n is amount of datapoints
    data = []
    x = scale*(np.random.random_sample((n, )) - 0.5) # this code generates an array of random numbers between -0.5 and 0.5, using random_sample from np, and then scales the array by the value of the scale variable

    for i in range(n):
        y = getY(x[i])
        data.append((x[i], y))

    return np.array(data)

data = sampleData() #  generating the data

# plotting the data
plt.rcParams["font.family"] = "monospace"
plt.grid(True)
plt.scatter(data[:, 0], data[:, 1], alpha = 0.5)
plt.title("Trained Data")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()
