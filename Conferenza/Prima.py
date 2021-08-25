

import numpy as np
import scipy as sp

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from Conferenza.functions import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # per lo split
from sklearn.preprocessing import StandardScaler        # per la normalizzazione



digits = datasets.load_digits()

X = digits.data[:, [2, 3]]
y = digits.target


# splitto i dati di iris in 30% testData e 70% trainingData
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# NORMALIZZAZIONE

sc = StandardScaler()
sc.fit(X_train)     # valuta media e sigma per standardizzare dopo
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print(X_test_std)
num = len(X_test_std)

svm = svm.SVC(gamma=0.001, C=1.)

# print(digits.target[:-num])
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(0, num))
plt.show()



