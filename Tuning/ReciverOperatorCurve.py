
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score      # auc = area under curve
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy import interp
import numpy as np


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)

X_train2 = X_train[:, [4, 14]]
cv = StratifiedKFold(random_state=1, n_splits=3)
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv.split(X_train, y_train)):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)'% (i+1, roc_auc))


plt.plot([0, 1], [0, 1], linestyle = '--', color = (0.6, 0.6, 0.6), label = 'random guessing')
mean_tpr /= cv.get_n_splits()
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label = 'mean ROC (area = %0.2f)' % mean_auc, lw = 2)
plt.plot([0, 0, 1], [0, 1, 1], lw = 2, linestyle = ':', color = 'black', label = 'perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.show()

pipe_lr.fit(X_train2, y_train)
y_pred2 = pipe_lr.predict(X_test[:, [4, 14]])

print('ROC AUC: %.3f' % (100*roc_auc_score(y_true=y_test, y_score=y_pred2)) + '%')
print('Accuracy: %.3f' % (100*accuracy_score(y_true=y_test, y_pred=y_pred2)) + '%')
