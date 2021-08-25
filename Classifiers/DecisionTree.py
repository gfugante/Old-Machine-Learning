
import numpy as np
from sklearn import datasets                            # per iris
from sklearn.model_selection import train_test_split    # per lo split
from Classifiers.functions import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target


# splitto i dati di iris in 30% testData e 70% trainingData
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# # TREE
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# tree.fit(X_train, y_train)
#
#
# plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.show()


# FOREST

forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()



