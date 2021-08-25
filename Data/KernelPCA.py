
# rbf kernel pca (gaussiana)
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import functions as f
import pandas as pd

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy import exp
import numpy as np



def rbf_kernel_pca(X, gamma, n_components):

    # calcolo le distanze reciproche e creo la matrice
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)

    # creo la matrice kernel simmetrica
    K = exp(-gamma * mat_sq_dists)

    # centro il kernel (è centrato lo spazio di partenza ma non quello di arrivo a priori)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # creo le coppie autovalore-autovettore (già ordinate in ordine dedrescente)
    eig_val, eig_vec = eigh(K)

    # seleziono le n maggiori (ultime n)
    alphas = np.column_stack((eig_vec[:, -i] for i in range(1, n_components+1)))

    lambdas = [eig_val[-i] for i in range(1, n_components+1)]

    return alphas, lambdas


# reprojection of a transformed data
def project_x(x_new_, X_, gamma_, alphas_, lambdas_):
    pair_dist = np.array([np.sum((x_new_ - row) ** 2) for row in X_])
    k = np.exp(-gamma_ * pair_dist)
    return k.dot(alphas_ / lambdas_)





X, y = make_moons(n_samples=100, random_state=123)
# colori diversi ma il pca è unsupervised!
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
# plt.show()

X_kpca, lambsas = rbf_kernel_pca(X, gamma=15, n_components=2)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
# ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1))+0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1))-0.02, color='blue', marker='o', alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# plt.show()


# con scikit
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
# plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
# plt.show()

# scikit_pca = PCA(n_components=2)
# X_spca = scikit_pca.fit_transform(X)
# fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
# ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
# ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()


X_kpca, lambdas = rbf_kernel_pca(X, gamma=3.3, n_components=2)
# fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
# ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
# ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()


lr = LogisticRegression()
X_train, X_test, y_train, y_test, X_combined, y_combined = f.prepare_data(X_kpca, y)

lr.fit(X_train, y_train)
f.plot_decision_regions(X_combined, y_combined, classifier=lr, test_idx=range(700, 1000))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()


# projecting new features with rbf_kernel_pca (da rivedere)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

x_new = X[25]
f.print_df(x_new)

x_proj = alphas[25]
f.print_df(x_proj)

x_reproj = project_x(x_new_=x_new, X_=X, gamma_=15, alphas_=alphas, lambdas_=lambdas)
f.print_df(x_reproj)


# KERNEL-PCA SCI-KIT

scikit_kpca = KernelPCA(n_components=2, gamma=3.5, kernel='rbf')
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


