
# Pipeline to build a procedure of processing data
# Stratified K-Fold Cross-Validation to tune the Logistic Regression (and the whole Pipeline)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from functions import print_df
import numpy as np

cancer = load_breast_cancer()
print_df(f"Cancer dataframe:\n\n{cancer}", start=True)

X = cancer.data
y = cancer.target
feature = cancer.feature_names

print_df(f"X [shape {X.shape}]:\n\n{X}")
print_df(f"y:\n\n{y}")
print_df(f"features:\n\n{feature}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print_df(f"N train = {len(y_train)}\n"
         f"N test = {len(y_test)}")


pipe_lr = Pipeline([("scl", StandardScaler()),
                    ("pca", PCA(n_components=2)),
                    ("lr", LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)

print_df('Test Accuracy: %.1f' % (100*pipe_lr.score(X_test, y_test)) + '%')


kfold = StratifiedKFold(n_splits=10, random_state=1)


scores = []
for k, (train, test) in enumerate(kfold.split(X_train, y_train)):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

print('-'*25 + '\n')
print("Cross validation accurcy: (%.1f+/-%.1f)" % (np.mean(scores)*100, np.std(scores)*100) + '%')
print('\n' + '-'*25)

# e come sempre si fa tutto più in fretta con scikit...

scores_new = cross_val_score(pipe_lr, X_train, y_train, cv=10, n_jobs=1)

print("\nNew scores:\n", scores_new)
print("\nCross validation accurcy: (%.1f+/-%.1f)" % (np.mean(scores_new)*100, np.std(scores_new)*100) + '%')
print('\n' + '-'*25)

