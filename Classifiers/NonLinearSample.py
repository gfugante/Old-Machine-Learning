
import numpy as np
import matplotlib.pyplot as plt
from Classifiers.functions import plot_decision_regions
from sklearn.svm import SVC


np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

svm = SVC(kernel='rbf', gamma=0.3, C=10.0)      # non linear kernel, Radial Basis Function (gaussianza con z = xi-xj)
svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
