
from sklearn import datasets                            # per iris
from sklearn.model_selection import train_test_split    # per lo split
from sklearn.preprocessing import StandardScaler        # per la normalizzazione
from sklearn.linear_model import Perceptron             # multiclass perceptron
from sklearn.metrics import accuracy_score              # accuracy (1-errors)
from Classifiers.functions import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

# print(np.unique(y))

# splitto i dati di iris in 30% testData e 70% trainingData
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print(len(X_train))
# print(y_test)

# NORMALIZZAZIONE

sc = StandardScaler()
sc.fit(X_train)     # valuta media e sigma per standardizzare dopo
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=0, max_iter=40)
ppn.fit(X_train_std, y_train)     # è multiclass, può fittarne più di uno alla volta

y_pred = ppn.predict(X_test_std)
miss = (y_test != y_pred).sum()
accuracy = (1 - miss/len(y_test))*100

print(f'Misclassified samples: {miss}')
print(f'Accuracy: {accuracy}%')
print(f'Accuracy sci-kit: %.1f' % (100*accuracy_score(y_test, y_pred)) + '%')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


