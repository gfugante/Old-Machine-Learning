import numpy as np
from sklearn.model_selection import GridSearchCV    # VEDI ANCHE RANDOMIZED GRID SEARCH CV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipe = Pipeline([('scl', StandardScaler()),
                 ('clf', SVC(random_state=1))])

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
              {'clf__C': param_range,
               'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs.fit(X_train, y_train)
print('\n', gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.1f' % (100*clf.score(X_test, y_test)) + '%')


scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)
print(scores)
print('Scoring accuracy SVM: %.1f +/- %.1f' % (100*np.mean(scores), 100*np.std(scores)) + '%')

# confronto con un tree

gs_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                       param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                       cv=5,
                       scoring='accuracy',
                       n_jobs=-1)

scores_tree = cross_val_score(gs_tree, X_train, y_train, scoring='accuracy', cv=5)
print(scores_tree)
print('Scoring accuracy tree: %.1f +/- %.1f' % (100*np.mean(scores_tree), 100*np.std(scores_tree)) + '%')




