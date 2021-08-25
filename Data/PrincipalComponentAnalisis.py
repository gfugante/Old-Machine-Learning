
# Vogliamo selezionare le features che contribuiscono maggiormente, per creare una nuova tabella che abbia k<d features.
# Per farlo usiamo il metodo PCA, creando la amtrice delle covarianze e individuando gli autovettori, selezionando i k
# autovettori che hanno corrispondenti autovalori più alti, di modo da scegliere le combinazioni lineari che hanno
# covarianza maggiore.
# Questo metodo è ovviamente molto sensibile a dati non standardizzati (o normalizzati).

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from functions import prepare_data, print_df, plot_decision_regions
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

X = wines.data
y = wines.target


#print_df(X, start=True)


X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined = prepare_data(X, y, random_state=2)

#print_df(X_train_std[:, 1])
#print_df(X_test_std[:, 1])


# creo la matrice delle covarianze, con autovalori e autovettori

cov_mat = np.cov(X_train_std.T)
eigen_val, eigen_vec = np.linalg.eig(cov_mat)

print('Matrice delle covarianze\n')
print(cov_mat.shape)
print_df(cov_mat)

print('Autovalori\n')
print_df(eigen_val)

print('Autovettori\n')
print(f'Shape: {eigen_vec.shape}\n')
print_df(eigen_vec[:, 0])

# variance explained ratio = lambdai/sum(lambdai)
# cumulative sum function
tot = sum(np.abs(eigen_val))
var_exp = [(i/tot) for i in sorted(eigen_val, reverse=True)]     # dal max al min
cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='upper left')
# plt.show()


# nota che gli autovettori sono le colonne non le righe
eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:, i]) for i in range(len(eigen_val))]
eigen_pairs.sort(reverse=True)

print('Pairs:\n')
print_df(eigen_pairs)

# creo la nuova matrice W con i primi due autovettori (matric projection)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W\n')
print_df(w)

# possiamo ora trasformare il dataframe in un nuovo data frame con sole due features
X_train_pca = np.dot(X_train_std, w)

# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m, edgecolors='black')
#
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.show()


# FACCIAMO ORA LA STESSA COSA CON SCI-KIT!
# ...e plottiamo le decisioni

pca = PCA(n_components=2)   # se setto su None ho poi la possibilità di vedere le percentuali con explained_variance_ratio
lr = LogisticRegression()

X_train_pca_new = pca.fit_transform(X_train_std)
X_test_pca_new = pca.transform(X_test_std)

lr.fit(X_train_pca_new, y_train)

idx = X_test_pca_new.shape[0]
print_df(idx)
X_combined_pca = np.vstack((X_train_pca_new, X_test_pca_new))

plot_decision_regions(X_combined_pca, y_combined, classifier=lr, test_idx=range(148, 178))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

