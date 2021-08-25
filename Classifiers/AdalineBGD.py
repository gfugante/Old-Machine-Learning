
# ADaptive LInear NEuron classifier
# GRADIENT BATCH DESCENT (minumum of the gradient with alla the wi updated
# with all samples instead of after of each sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classifiers.classes import AdalineBGD
from Classifiers.functions import plot_decision_regions

# creo il data frame'
df = pd.read_csv('dataTest.csv')


y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values


# # double plot of costs
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
#
# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline - Learning rate 0.01')
#
# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Sum-squared-error')
# ax[1].set_title('Adaline - Learning rate 0.0001')
#
# plt.show()

# I COSTI SONO DIVERSI PER DIVERSI LEARNNG-RATES ETA!
# per il gradient-descending si possono normalizzare i dati per rendere indipendente da eta il costo


# NORMALIZZAZIONE

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# ETA = 0.01

ada = AdalineBGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
