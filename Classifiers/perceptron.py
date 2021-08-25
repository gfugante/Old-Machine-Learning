import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classifiers.classes import Perceptron
from Classifiers.functions import plot_decision_regions


df = pd.read_csv('dataTest.csv')

# print(df.tail())

y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# plot dataTest
plt.scatter(X[:49, 0], X[:49, 1], color='red', marker='o', label='setosa')
plt.scatter(X[49:99, 0], X[49:99, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.01,  n_iter=10)
ppn.fit(X, y)

# plot errors/epoches
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of errors')
plt.show()

# plot decision areas
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# L'ALGORITMO ORA SA DECIDERE CHE FIORE è IN BASE ALL'AREA SOPRA DEFINITA!
