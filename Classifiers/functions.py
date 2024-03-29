import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier,
                       test_idx=None, resolution=0.02):
       # setup marker generator and color map
       markers = ('s', 'x', 'o', '^', 'v')
       colors = ('red', 'blue', 'lightgreen', 'cyan', 'grey')
       cmap = ListedColormap(colors[:len(np.unique(y))])
       s=20
       # plot the decision surface
       x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
       x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
       xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
       Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
       Z = Z.reshape(xx1.shape)
       plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
       plt.xlim(xx1.min(), xx1.max())
       plt.ylim(xx2.min(), xx2.max())



       for idx, cl in enumerate(np.unique(y)):
           plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                       alpha=1.0, c=cmap(idx),
                       marker=markers[2], label=cl, s=s)

        #   hilights test data
       if test_idx:
           X_test, y_test = X[test_idx, :], y[test_idx]
           mark = 'o'

           plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                       alpha=1.0, linewidth=1, marker='o', edgecolor='black',
                       s=s, label='test set')


# SIGMOIDE
def sigmoid(z, m=1.0):
    return 1.0/(1.0 + np.exp(-z*m))


# # test sigmoide
# z = np.arange(-7, 7, 0.1)
# phi_z = sigmoid(z, 10)
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k')
# plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
# plt.axhline(y=0.0, ls='dotted', color='k')
# plt.axhline(y=0.5, ls='dotted', color='k')
# plt.axhline(y=1.0, ls='dotted', color='k')
# plt.yticks([0.0, 0.5, 1.0])
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')
# plt.show()


def space():
    print()
    print('-'*23)
    print()



