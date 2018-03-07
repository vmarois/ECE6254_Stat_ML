"""
* Python script for the problem 1 of Homework #3.
* @author Vincent MAROIS vmarois3@gatech.edu
* @date 26/02/2018
* @version 1.0
"""
import numpy as np
from sklearn.utils import shuffle

# dataset:
X = np.array([
    # class 1
    [1, 0, 0, 1],  # simply inverted first 2 elements: takes 21 iterations instead of 29.
    [0, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],

    # class 2
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    [1, 1, 1, 1]
])

# labels:
Y = np.array([1, 1, 1, 1,
              -1, -1, -1, -1])

# try shuffling the dataset
X2, Y2 = shuffle(X, Y)
# 21 iterations seems to be the optimal


def perceptron(X, Y):
    # initial weights
    w = np.array([-1, -2, -2, 0])

    epochs = 10

    # count number of iterations
    nb_iter = 0

    # counter to check if 8 iterations without update
    no_update = 0
    outer_break = False  # break out 2 nested for loops

    for t in range(epochs):

        for i, x in enumerate(X):
            nb_iter += 1
            print('####### iter :', nb_iter)
            print('Data point chosen : (', X[i], ', ', Y[i], ')')
            print('Current weights w : ', w, '\n')

            if (np.dot(w, X[i]) * Y[i]) <= 0:
                print(' We need to update the weights.')
                w = w + X[i]*Y[i]
                no_update = 0
                print(' New weights : ', w, '\n')
            else:
                print(' No update necessary.\n')
                no_update += 1

            # check if convergence
            if no_update >= 8:
                print('No update for the entire dataset: Convergence.')
                print('Number of iterations taken : ', nb_iter)
                outer_break = True
                break
        if outer_break:
            break
    return w


w = perceptron(X, Y)
print("\nFinal weights : ", w)
print("\nSign of the predictions using the current weights : ", np.sign(np.dot(X, w)))
