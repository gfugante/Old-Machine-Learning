
# NORMALIZZAZIONE E STANDARDIZZAZIONE
# ottimizzano tutte e due le performance degli algoritmi,
# la prima è preferibile per algoritmi tipo KNN e Adaline o dove abbiamo problemi di norme,
# mentre la seconda è utile per algoritmi che inizializzano i pesi a 0 o numeri random vicini
# a zero, come il Logistic Regression e SVM

# Xnorm = (xi-xmin)/(xmax-xmin)
# Xstd = (xi-mu)/sigma

# la standardizzazione inoltre mantiene informazioni sui dati anomali


from sklearn.preprocessing import MinMaxScaler      # normalizzazione
from sklearn.preprocessing import StandardScaler    # standardizzazione
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from functions import print_df


wines = load_wine()

X, y = wines.data, wines.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# NORMALIZZAZIONE
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# STANDARDIZZAZIONE
stsc = StandardScaler()
X_train_std = stsc.fit_transform(X_train)
X_test_std = stsc.transform(X_test)


# nel primo necessariamente deve esserci un 1. e un 0. e nessun valore negativo,
# nel secondo non ci sono restrizioni ma è distribuito gaussianamente

print_df(X_train_norm[:, 1], start=True)
print_df(X_train_std[:, 1])




