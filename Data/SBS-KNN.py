
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from classes import SBS
from sklearn.datasets import load_wine
from functions import prepare_data


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


X, y = wines.data, wines.target

X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined = prepare_data(X, y, random_state=0)


knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[7])
print(k5)

for i in k5:
    if i != 0:
        print(wines.columns[i])

# original dataset
print('-'*25)
knn.fit(X_train_std, y_train)
print('Training accuracy: %.2f' % (100*knn.score(X_train_std, y_train)) + '%')
print('Test accuracy: %.2f' % (100*knn.score(X_test_std, y_test)) + '%')

# new dataset
print('-'*25)
knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy: %.2f' % (100*knn.score(X_train_std[:, k5], y_train)) + '%')
print('Test accuracy: %.2f' % (100*knn.score(X_test_std[:, k5], y_test)) + '%')
print('-'*25)