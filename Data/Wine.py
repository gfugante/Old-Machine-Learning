

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
import functions as f

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

f.print_df(X.shape, start=True)

X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined = f.prepare_data(X,y)


f.print_df(np.unique(y_train))
f.print_df(np.unique(y_test))


lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)

print(f'Training accuracy: %.2f' % (100*lr.score(X_train_std, y_train)) + '%')
print(f'Test accuracy: %.2f' % (100*lr.score(X_test_std, y_test)) + '%')


f.print_df(lr.intercept_)
f.print_df(lr.coef_)


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.55, 0.75])

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10.0**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0 ** c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], color=color, label=wines.columns[column+1])


plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()
