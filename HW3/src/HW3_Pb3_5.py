"""
* Python script for the problem 5 of Homework #3.
* @author Vincent MAROIS vmarois3@gatech.edu
* @date 27/02/2018
* @version 1.0
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from math import floor

mnist = fetch_mldata('MNIST original')  # This dataset contains 70000 handwritten digits, each of size 28 × 28 pixels.

X = mnist.data
y = mnist.target

# We only build a classifier to classify between the images of the digits "4" and "9"
X4 = X[y == 4, :]
y4 = y[y == 4]
X9 = X[y == 9, :]
y9 = y[y == 9]

# split into training & test set : training sets have 4000 instances each
test_size_4 = 1 - 4000/(X4.shape[0])
test_size_9 = 1 - 4000/(X9.shape[0])
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=test_size_4)
X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, test_size=test_size_9)

# concatenate the datasets into 1 to train the SVM
# doing so ensure the classes are well-balanced in term of number of training instances
X_train = np.concatenate((X4_train, X9_train), axis=0)
y_train = np.concatenate((y4_train, y9_train), axis=0)

X_test = np.concatenate((X4_test, X9_test), axis=0)  # 2824 instances of class 'digit = 4' & 2958 for class 'digit = 9'
y_test = np.concatenate((y4_test, y9_test), axis=0)


# 1/ Train the SVM using the linear and quadratic kernel
def svm_poly_kernel():
    """
    Using the entire training set, we set a 10-fold stratified cross-validation to compute the test score :
    i.e we divide the dataset into 10 folds, train on 9 folds and test on the last one. We then select the value of p &
    C for which this test score was the highest.
    """
    result = {'linear': {}, 'quadratic': {}}

    # define the range of values to try for C
    C = np.logspace(-5, 2, 8)
    param_grid = {"C": C}

    for degree in [1, 2]:

        # instantiate the GridSearch object & fit it on the training dataset
        clf = SVC(kernel='poly', degree=degree)
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, n_jobs=-1)  # using 10-fold cross-validation
        print('Performing a grid search on C for the {} kernel'.format('linear' if degree == 1 else 'quadratic'))
        grid_search.fit(X_train, y_train)

        print('Optimal C value for the {} kernel : {}'.format('linear' if degree == 1 else 'quadratic',
                                                              grid_search.best_params_['C']))

        # retrain the SVM using the optimal value for C on the entire training set & score on the test set:
        clf = SVC(C=grid_search.best_params_['C'], kernel='poly', degree=degree)
        clf.fit(X_train, y_train)

        # calculate the probability of error on test set
        Pe = 1 - clf.score(X_test, y_test)
        print('Probability of error on the test set : {}'.format(Pe))

        # get number of support vectors
        print('Number of support vectors : {}'.format(clf.support_vectors_.shape[0]))
        if degree == 1:
            result['linear'] = {"C": grid_search.best_params_['C'],
                                "sv": clf.support_vectors_.shape[0]}
        else:
            result['quadratic'] = {"C": grid_search.best_params_['C'],
                                   "sv": clf.support_vectors_.shape[0]}
    return result


def create_plots_poly_kernel():
    """
    Create a plot showing the evolution of the test error for varying values of C for both the linear & quadratic kernel
    """
    C = np.logspace(-7, 3, 11)

    for d in [1, 2]:
        error_C = []
        for c in C:
            print('C = ', c)
            clf = SVC(C=c, kernel='poly', degree=d)
            clf.fit(X_train, y_train)
            error_C.append(1 - clf.score(X_test, y_test))
        plt.style.use('ggplot')
        plt.semilogx(C, error_C)
        plt.title(r'Test error of the {} kernel SVM wrt C'.format('linear' if d == 1 else 'quadratic'))
        plt.xlabel('C')
        plt.ylabel('Test error Pe')
        plt.savefig('test_error_{}_C.png'.format('lin' if d == 1 else 'quad'))
        plt.clf()


# 2/ Train the SVM using the rbf kernel
def svm_rbf_kernel():
    """
    We perform the same type of grid search to determine the best values for C & gamma.
    We can treat C & gamma independently: we search the best value of C first, then set it constant and do the same for
    gamma.
    """
    # define the range of values to try for C & gamma
    C = np.logspace(-5, 2, 8)
    gamma = np.logspace(-8, 0, 9)

    # instantiate the GridSearch object & fit it on the training dataset, to search for the optimal C
    clf = SVC(kernel='rbf')  # default for gamma is 1/n_features
    print('Performing a grid search on C for the rbf kernel')
    grid_search = GridSearchCV(clf, param_grid={"C": C}, cv=10, n_jobs=-1)  # using 10-fold cross-validation
    grid_search.fit(X_train, y_train)

    # get the optimal C
    print('Optimal C value for the rbf kernel : {}'.format(grid_search.best_params_['C']))
    opt_C = grid_search.best_params_['C']

    # Perform the same task for gamma
    clf = SVC(C=opt_C, kernel='rbf')
    print('Performing a grid search on gamma for the rbf kernel (C is fixed to {})'.format(opt_C))
    grid_search = GridSearchCV(clf, param_grid={"gamma": gamma}, cv=10, n_jobs=-1)  # using 10-fold cross-validation
    grid_search.fit(X_train, y_train)

    # get the optimal gamma
    print('Optimal gamma value for the rbf kernel : {}'.format(grid_search.best_params_['gamma']))
    opt_gamma = grid_search.best_params_['gamma']

    # retrain the SVM using the optimal values for C & gamma on the entire training set & score on the test set:
    clf = SVC(C=opt_C, kernel='rbf', gamma=opt_gamma)
    clf.fit(X_train, y_train)

    # calculate the probability of error
    Pe = 1 - clf.score(X_test, y_test)
    print('Probability of error on the test set : {}'.format(Pe))

    # get number of support vectors
    print('Number of support vectors : {}'.format(clf.support_vectors_.shape[0]))

    return {'rbf': {"C": opt_C,
                    "gamma": opt_gamma,
                    "sv": clf.support_vectors_.shape[0]}}


def create_plots_rbf_kernel():
    """
    Create 2 plots showing the evolution of the test error for varying values of C and gamma. Produce 2 plots as we
    consider C & gamma independently (we set one parameter to its optimal value & vary the other).
    """
    C = np.logspace(-4, 2, 7)
    gamma = np.logspace(-8, 0, 9)

    opt_C = 1
    opt_gamma = 1e-6

    error_C = []
    for c in C:
        print('C = ', c)
        clf = SVC(C=c, kernel='rbf', gamma=opt_gamma)
        clf.fit(X_train, y_train)
        # calculate the probability of error
        error_C.append(1 - clf.score(X_test, y_test))
    plt.style.use('ggplot')
    plt.semilogx(C, error_C)
    plt.title(r'Test error of the RBF kernel SVM wrt C ($\gamma$ = {})'.format(opt_gamma))
    plt.xlabel('C')
    plt.ylabel('Test error Pe')
    plt.savefig('test_error_rbf_C.png')
    plt.clf()

    error_gamma = []
    for g in gamma:
        print('gamma = ', g)
        clf = SVC(C=opt_C, kernel='rbf', gamma=g)
        clf.fit(X_train, y_train)
        # calculate the probability of error
        error_gamma.append(1 - clf.score(X_test, y_test))
    plt.style.use('ggplot')
    plt.semilogx(gamma, error_gamma)
    plt.title(r'Test error of the RBF kernel SVM wrt $\gamma$ (C = {})'.format(opt_C))
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Test error Pe')
    plt.savefig('test_error_rbf_gamma.png')


# 3/ Plot for each kernel the 16 support vectors that violate the margin by the greatest amount
def support_vectors_violation(clf):
    """
    Turns in a 4 × 4 subplot showing images of the 16 support vectors that violate the margin by the greatest amount.
    We're referring to support vectors in the training set that violate the margin, irrespective of how they're being
    classified. These will be the hardest examples to classify.
    :param clf: sklearn.svm.SVC() object with the correct parameters
    :return: 4x4 plot of support vectors that violate the margin by the greatest amount
    """
    # start by training the input SVM on the training dataset
    clf.fit(X_train, y_train)

    # get the support vectors of the training set & their label
    sv = clf.support_vectors_
    sv_labels = y_train[clf.support_]

    # get distance of support vectors to the separating hyperplane
    sv_distance = clf.decision_function(sv)  # looks like decision < 0 for class '4' & > 0 for class '9'
    # same shape as sv_labels -> n x 1

    # Determine the distance by which the support vectors violate the margin hyperplanes

    # This distance is :
    # - 0 for correctly classified instances (that don't violate the margin hyperplane),
    # - in [0, 1) for correctly classified instances that violate the margin hyperplane
    # - > 1 for misclassified instances that don't violate the outermost margin hyperplane
    # - > 2 for misclassified instances that violate the outermost margin hyperplane

    # => The bigger this distance, the more the instance violates the hyperplane

    # create arrays of 0
    dist_from_margin = np.zeros_like(sv_labels)

    for i, support_vector in enumerate(sv):  # loop over the support vectors

        # the instance is properly classified
        if clf.predict(support_vector.reshape(1, -1)) == sv_labels[i]:

            # the instance doesn't violate the margin hyperplane
            if np.abs(sv_distance[i]) >= 1:
                dist_from_margin[i] = 0

            # else the instance violates the margin hyperplane
            elif np.abs(sv_distance[i]) < 1:
                dist_from_margin[i] = 1 - np.abs(sv_distance[i])

        # else, the instance is misclassified, meaning the distance from the separating hyperplane is of the wrong sign
        # the distance from the (correct) margin hyperplane is 1 + abs(distance from separating hyperplane)
        elif not clf.predict(support_vector.reshape(1, -1)) == sv_labels[i]:
            dist_from_margin[i] = 1 + np.abs(sv_distance[i])
    
    # find the index of the 16 largest values of dist_from_margin, i.e the 16 support vectors that violate the margin
    # by the greatest amount
    idx = np.argpartition(dist_from_margin, -16)[-16:]
    print("Distance by which the 16 'worst' support vectors violate the margin: \n", dist_from_margin[idx])
    print('\nPredicted labels :\n', clf.predict(sv[idx]))
    print('\nTrue labels :\n', sv_labels[idx])

    # keep only these 16 support vectors & their true label
    sv = sv[idx]
    sv_labels = sv_labels[idx]

    # create plot
    f, axarr = plt.subplots(4, 4, figsize=(7, 7))
    for i, digit in enumerate(sv):
        axarr[floor(i / 4), floor(i % 4)].imshow(digit.reshape((28, 28)), cmap='gray')
        axarr[floor(i / 4), floor(i % 4)].axis('off')
        axarr[floor(i / 4), floor(i % 4)].set_title('{label}'.format(label=int(sv_labels[i])))
    plt.savefig('support_vectors_margin_violation.png')


if __name__ == '__main__':

    # grid search & test error plot for the polynomial kernels
    poly_result = svm_poly_kernel()
    print(poly_result)
    create_plots_poly_kernel()

    # grid search & test error plot for the rbf kernel
    rbf_result = svm_rbf_kernel()
    print(rbf_result)
    create_plots_rbf_kernel()

    # plot the support vectors that most violate the margin for the polynomial kernels
    poly_result = {'linear': {'C': 0.001, 'sv': 853}, 'quadratic': {'C': 0.0001, 'sv': 673}}
    lin_clf = SVC(C=poly_result['linear']['C'], kernel='poly', degree=1)
    quad_clf = SVC(C=poly_result['quadratic']['C'], kernel='poly', degree=2)

    support_vectors_violation(lin_clf)
    support_vectors_violation(quad_clf)

    # plot the support vectors that most violate the margin for the rbf kernel
    rbf_clf = SVC(C=1, kernel='rbf', gamma=1e-6)
    support_vectors_violation(rbf_clf)
    print('Done.')
