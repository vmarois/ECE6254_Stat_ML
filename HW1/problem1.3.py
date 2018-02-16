import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import GridSearchCV
np.random.seed(2017)  # Set random seed so results are repeatable

n = [100, 500, 1000, 5000]  # number of training points

plt.figure(figsize=(8, 8))

for idx, size in enumerate(n):

    # generate a simple 2D dataset
    X, y = datasets.make_moons(size, 'True', 0.3)

    # perform a grid search on n_neighbors :
    # using the entire dataset, we set a 10-fold stratified cross-validation to compute the test score :
    # i.e we divide the dataset into 10 folds while preserving the percentage of samples for each class, train on 9
    # folds and test on the last one. We then select the value of k for which this test score was the highest.
    # (quantitative approach).

    # define the range of values to try for k
    n_neighbors = np.arange(1, 21)
    param_grid = {"n_neighbors": n_neighbors}
    print('Performing grid search using the following parameters & ranges: \n', param_grid)

    # instantiate the GridSearch object & fit it on the dataset (might take some time for higher values of n)
    grid_search = GridSearchCV(neighbors.KNeighborsClassifier(weights='uniform', n_jobs=4), param_grid=param_grid,
                               cv=10) # using 10-fold cross-validation to detect the k value
    grid_search.fit(X, y)

    # print some info
    print('Optimal k value for n = {} data points: {}'.format(size, grid_search.best_params_['n_neighbors']))
    print('Best score: {}\n'.format(grid_search.best_score_))

    # Create instance of KNN classifier with the best value of k
    classifier = neighbors.KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], weights='uniform')
    classifier.fit(X, y)

    # plot the decision boundary.
    # begin by creating the mesh [x_min, x_max]x[y_min, y_max].
    h = .02  # step size in the mesh
    x_delta = (X[:, 0].max() - X[:, 0].min())*0.05  # add 5% white space to border
    y_delta = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - x_delta, X[:, 0].max() + x_delta
    y_min, y_max = X[:, 1].min() - y_delta, X[:, 1].max() + y_delta
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.subplot(2, 2, idx+1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("%i-NN on %i data points" % (grid_search.best_params_['n_neighbors'], size))

# show the plot
plt.savefig('grid_search_knn.png')
plt.show()

