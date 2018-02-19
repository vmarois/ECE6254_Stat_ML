import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import seaborn as sns
import time


# the logistic function
def logistic_func(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    # split into positive and negative to improve stability
    g[t>=0.0] = 1.0 / (1.0 + np.exp(-t[t>=0.0])) 
    g[t<0.0] = np.exp(t[t<0.0]) / (np.exp(t[t<0.0])+1.0)
    return g


# function to compute log-likelihood
def neg_log_like(theta, x, y):
    g = logistic_func(theta,x)
    return -sum(np.log(g[y>0.5])) - sum(np.log(1-g[y<0.5]))


# function to compute the gradient of the negative log-likelihood
def log_grad(theta, x, y):
    g = logistic_func(theta,x)
    return -x.T.dot(y-g)


# function to compute the hessian matrix
def hessian(theta, x):
    """
    Compute the Hessian matrix of the log-likelihood, as H = (X.T).D.X, where D = diag( g(theta*x)(1-g(theta*x)) )
    Instead of using a one-liner like 'x.dot(x.T).dot(g).dot(np.ones(len(g))-g)', we break it into the intermediate
    matrix multiplications to avoid dealing with very large sparse matrices. Speed computations up.
    :param theta: the parameters (b, w) of the logistic regression.
    :param x: the training instances
    :return: The hessian matrix H (3x3 matrix)
    """
    g = logistic_func(theta, x)
    D = np.multiply(g, np.ones(len(g)) - g)
    D = np.diag(D)
    L = x.T.dot(D)
    H = L.dot(x)
    return H


# implementation of the newton method for logistic regression
def newton_method(theta, x, y, tol, maxiter):
    """
    Implementation of Newton's method for the logistic regression solver.
    We compute the Hessian matrix of the log-likelihood at each iteration to guide the gradient descent.
    Should work well given that the log-likelihood is a convex function.
    :param theta: the 3 parameters defining the 2D classification boundary for the logistic regression.
    :param x: training instances.
    :param y: classification labels associated to the training instances.
    :param tol: the tolerance threshold to estimate when the gradient is vanishing (we should then have theta(n) ~theta(n-1))
    :param maxiter: maximum number of iterations allowed.
    :return:
    """
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0 * tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        theta = theta - (np.linalg.inv(hessian(theta, x))).dot(log_grad(theta, x, y))
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2] - nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec), iter

    
# implementation of gradient descent for logistic regression
def grad_desc(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        theta = theta - (alpha * log_grad(theta, x, y)) 
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec), iter


# implementation of stochastic gradient descent for logistic regression
def stoc_grad_desc(theta, x, y, alpha, tol, maxiter):

    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):

        idx = np.random.randint(0, len(x))

        theta = theta - (alpha * log_grad(theta, np.asarray(x[idx]), np.asarray(y[idx])))
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec), iter


# function to compute output of LR classifier
def lr_predict(theta,x):
    # form Xtilde for prediction
    shape = x.shape
    Xtilde = np.zeros((shape[0],shape[1]+1))
    Xtilde[:,0] = np.ones(shape[0])
    Xtilde[:,1:] = x
    return logistic_func(theta,Xtilde)


# Generate dataset
np.random.seed(2017)  # Set random seed so results are repeatable
x,y = datasets.make_blobs(n_samples=100000, n_features=2, centers=2, cluster_std=6.0)

# build classifier
# form Xtilde
shape = x.shape
xtilde = np.zeros((shape[0],shape[1]+1))
xtilde[:,0] = np.ones(shape[0])
xtilde[:,1:] = x

# Initialize theta to zero
theta = np.zeros(shape[1]+1)

# Run gradient descent
alpha = 1e-6
tol = 1e-3
MAXITER = 10000

# Choose one of the methods as the solver for logistic regression.
#theta, cost, iter = grad_desc(theta, xtilde, y, alpha, tol, MAXITER)
#theta, cost, iter = newton_method(theta, xtilde, y, tol, MAXITER)
#theta, cost, iter = stoc_grad_desc(theta, xtilde, y, alpha, tol, MAXITER)
#########################


def question1(theta, xtilde, y):
    """
    Study the impact of alpha & tol on the number of iterations before convergence. Keep the other parameter constant
    when varying one.
    Saves a plot of iterations vs alpha & tol to file.
    """

    # define criteria as a logspace between 10 ** -6 & 1
    alpha = np.logspace(-6,0, 12)
    tol = np.logspace(-6, 0, 12)

    # list to store the number of iterations
    iter_alpha = []
    iter_tol = []

    for a in alpha:
        _, _, iter = grad_desc(theta, xtilde, y, a, 0.1e-3, MAXITER)
        iter_alpha.append(iter)

    for t in tol:
        _, _, iter = grad_desc(theta, xtilde, y, 0.001, t, MAXITER)
        iter_tol.append(iter)

    plt.figure()
    plt.subplot(121)
    plt.semilogx(alpha, iter_alpha)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Number of iterations')

    plt.subplot(122)
    plt.semilogx(tol, iter_tol)
    plt.xlabel('tol')
    plt.ylabel('Number of iterations')

    plt.savefig('Question_2_2_1.png')
    plt.show()


def methods_comparison(theta, xtilde, y, tol, nb_test=30, verbose=False):
    """
    Compare the run time & number of iterations for all three algorithms when applied to the dataset.
    To obtain more accurate results, we average the measurements over several executions.
    :param nb_test: the number of executions taken in consideration to average the results
    :return: print info on runtime & # of iterations of 3 solvers for a given dataset size.
    """

    print('Size of the dataset: {} data points'.format(len(y)))
    # define lists to store the run time & number of iterations of each method
    iter_grad_desc = []
    time_grad_desc = []

    iter_newton_method = []
    time_newton_method = []

    iter_stoc_grad_desc = []
    time_stoc_grad_desc = []

    for i in range(nb_test):
        if verbose:
            print('\nRun : ', i)

        start_time = time.time()
        _, _, iter1 = grad_desc(theta, xtilde, y, alpha, tol, MAXITER)
        time_grad_desc.append(time.time() - start_time)
        iter_grad_desc.append(iter1)
        if verbose:
            print('Time needed to perform Gradient Descent : ', time.time() - start_time)
            print('Number of iterations needed : ', iter1)

        start_time = time.time()
        _, _, iter2 = newton_method(theta, xtilde, y,tol, MAXITER)
        time_newton_method.append(time.time() - start_time)
        iter_newton_method.append(iter2)
        if verbose:
            print("Time needed to perform Newton's method : ", time.time() - start_time)
            print('Number of iterations needed : ', iter2)

        start_time = time.time()
        _, _, iter3 = stoc_grad_desc(theta, xtilde, y, alpha, tol, MAXITER)
        time_stoc_grad_desc.append(time.time() - start_time)
        iter_stoc_grad_desc.append(iter3)
        if verbose:
            print('Time needed to perform Stochastic Gradient descent : ', time.time() - start_time)
            print('Number of iterations needed : ', iter3)

    print('\nAverage number of iterations for Gradient Descent : ', np.mean(iter_grad_desc))
    print('Average run time for Gradient Descent : ', np.mean(time_grad_desc))

    print("\nAverage number of iterations for Newton's method : ", np.mean(iter_newton_method))
    print("Average run time for Newton's method : ", np.mean(time_newton_method))

    print('\nAverage number of iterations for Stochastic Gradient Descent : ', np.mean(iter_stoc_grad_desc))
    print('Average run time for for Stochastic Gradient Descent : ', np.mean(time_stoc_grad_desc))

    """
    # create a plot for the distribution of the number of iterations for the Stochastic Gradient Descent
    plt.figure()
    sns.distplot(iter_stoc_grad_desc, kde=True)
    plt.title('Distribution of the number of iterations for the Stochastic Gradient Descent')
    plt.savefig('dist_iter_stoc_grad_desc.png')
    plt.show()
    """

"""
# Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_delta = (x[:, 0].max() - x[:, 0].min())*0.05 # add 5% white space to border
y_delta = (x[:, 1].max() - x[:, 1].min())*0.05
x_min, x_max = x[:, 0].min() - x_delta, x[:, 0].max() + x_delta
y_min, y_max = x[:, 1].min() - y_delta, x[:, 1].max() + y_delta
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = lr_predict(theta,np.c_[xx.ravel(), yy.ravel()])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

# Show the plot
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Logistic regression classifier")
plt.show()
"""

if __name__ == '__main__':
    #question1(theta, xtilde, y)
    methods_comparison(theta, xtilde, y, tol, nb_test=1, verbose=False)  # set nb_test = 1 if n = 1e6 or larger
    print('done')
