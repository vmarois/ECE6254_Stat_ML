import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def plot_histogram(beta=3):
    """
    Create an histogram for the random variable Z = max Xi, i=1..m, where Xi ~ Normal(0,1).
    m = 10 ** beta
    :param beta: used to compute m.
    :return: A plt.hist + saved to file.
    """
    m = 10 ** beta

    # generate m normal random variables of 100 points each
    X = np.random.randn(100, m)

    # take the maximum along the rows
    Z = np.max(X, axis=1)

    # plot the pdf with a gaussian kernel density estimate
    plt.subplot(121)
    sns.distplot(Z, kde=True)
    plt.title(r'Histogram of Z for $\beta$ = {}'.format(beta))

    # plot the cdf and find t in relation with Q3)
    plt.subplot(122)
    plt.hist(Z, bins=25, normed=True, cumulative=True)
    plt.title(r'P[Z $\leq$ t]$\geq$0.9 for t$\geq$%0.4f' % (np.sqrt(2*(np.log(m) + np.log(10)))))

    print('P[Z <= t] >= 0.9 for t >= %0.4f using the inverse cdf' % (norm.ppf(0.9 ** (1/m))))
    print('P[Z <= t] >= 0.9 for t >= %0.4f using the Chernoff bounding method'
          % (np.sqrt(2*(np.log(m) + np.log(10)))))

    # save the plot to file & show the plot
    plt.savefig('histogram_beta_{}.png'.format(beta))

    plt.show()


if __name__ == '__main__':
    plot_histogram(beta=3)
