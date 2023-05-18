import numpy as np
import random
from Neuron import Neuron
from sklearn.model_selection import train_test_split


def main():
    data = np.loadtxt("data.csv", delimiter=",")
    data = np.hstack((np.ones(shape=[data.shape[0], 1]), data))

    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # weights = np.zeros(shape=[X.shape[1], 1])
    # weights[0] = 1
    # np.array([[0.43114389], [0.47681936], [-1.2202026]])
    weights = np.array([[0.24606208], [1.22437169], [-2.28076799]])

    neuron = Neuron(weights)

    neuron.SGD(X_train, y_train, batch_size=int(X_train.shape[0] / 50), learning_rate=0.01, max_steps=30000)
    print("weights: " + str(neuron.w))


if __name__ == "__main__":
    main()