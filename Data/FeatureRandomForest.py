
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


wines = load_wine()
wines.columns = ['Class label', 'Alcohol',
                'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids',
                'Nonflavanoid phenols',
                'Proanthocyanins',
                'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines',
                'Proline']

features = wines.columns[1:]
X, y = wines.data, wines.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

# valuto quanto influiscono le features in base ai pesi imparati dal forest (lento ma efficace)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]     # reverse perchè argsort ordina dal min al max


for f in range(X_train.shape[1]):
    print("%2d) %-*s %.2f" % (f + 1, 30, features[f], 100*importances[indices[f]]) + '%')

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), features, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


