
import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined = prepare_data(X, y, random_state=0)
print_df(X_train_std, start=True)
print_df(np.unique(y_train))
np.set_printoptions(precision=4)
mean_vecs = []

for label in range(0, 3):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label+1, mean_vecs[label]))

# print_df(mean_vecs)

# Sw: within Scatter matrix = sum(Si)
# Si: individual scatter matrix = sum( (mean_veci-x)(mean_veci-x)T )

# number of features
d = 13
print_df(d)

S_W = np.zeros((d, d))
for label, mv in zip(range(0, 3), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += np.dot((row-mv), (row-mv).T)
    S_W += class_scatter

print_df('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
print_df(f'Last inner-scatter matrix:\n\n {class_scatter}')
print_df(f'Within-scatter matrix:\n\n {S_W}')

print_df(f'Class label distributions: {np.bincount(y_train)}')


# le distribuzioni non sono uniformi! bisogna normalizzare le Si prima di sommarle a Sw
# LA MATRICE DELLE COVARIANZE è GIà UNA VERSIONE NORMALIZZATA DELLE Si!

S_W = np.zeros((d, d))
for label, mv in zip(range(0, 3), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

print_df(f'Scaled within-scatter matrix: \n\n {S_W}')


# Sb Between-scatter matrix: sum( Ni(mi-m)(mi-m) )
# dove mi sono i vettori delle medie e m le edie calcolai su tutto X_train_std

mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    N = X_train_std.shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    S_B += N * np.dot((mean_vec-mean_overall), (mean_vec-mean_overall).T)

print_df(f'Between-scatter matrix: \n\n {S_B}')

# troviamo gli autovalori e autovettori dalla matrice Sw-1*Sb

eigen_val, eigen_vec = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.real(np.abs(eigen_val[i])), eigen_vec[:, i]) for i in range(len(eigen_val))]

print_df(f'Eigen values:\n\n {np.real(eigen_val)}')


eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print_df(f'Eigen pairs:\n\n {eigen_pairs}')


tot = sum(eigen_val.real)
discr = [(i / tot) for i in sorted(eigen_val.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([0., 1.1])
plt.legend(loc='best')
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print_df(f'Matrix W:\n\n{w}')

# costruisco la nuova matrice delle 2 features
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1], c=c, marker=m, label=l, edgecolors='black')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()

# CON SCI-KIT

lda = LinearDiscriminantAnalysis(n_components=2)
lr = LogisticRegression(C=100000000)

X_train_lda_new = lda.fit_transform(X_train_std, y_train)
X_test_lda_new = lda.transform(X_test_std)
X_combined_lda_new = np.vstack((X_train_lda_new, X_test_lda_new))

print_df(X_train_lda_new.shape)

lr.fit(X_train_lda_new, y_train)


plot_decision_regions(X_combined_lda_new, y_combined, classifier=lr, test_idx=range(148, 178))
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower left')
plt.show()




