
# matrice quadrata che riporta i conteggi dei true positive, true negative, false positive, false negative di
# un classificatore. Rappresenza in sostanza la performance del classificatore.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score,recall_score, f1_score
from functions import print_df, plot_decision_regions
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
pipe_svc = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(random_state=1))])

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(4., 4.))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()


print_df('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred), start=True)
print_df('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print_df('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

