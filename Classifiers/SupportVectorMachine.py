
# Support Vector Machine
# C parametro di quanto sono generoso (C grande = margine piccolo)

from sklearn.svm import SVC

import numpy as np
from sklearn import datasets                            # per iris
from sklearn.model_selection import train_test_split    # per lo split
from sklearn.preprocessing import StandardScaler        # per la normalizzazione
from Classifiers.functions import plot_decision_regions
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target


# splitto i dati di iris in 30% testData e 70% trainingData
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# NORMALIZZAZIONE

sc = StandardScaler()
sc.fit(X_train)     # valuta media e sigma per standardizzare dopo
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


svm = SVC(kernel='rbf', C=10.0, gamma=250.0, random_state=0)
svm.fit(X_combined_std, y_combined)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
