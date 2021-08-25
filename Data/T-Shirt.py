
import numpy as np
import pandas as pd
import functions as f
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']


f.print_df(lines=31, data_frame=df, start=True)


# manual mapping ordinal features
size_map = {'XL': 3, 'L': 2, 'M': 1, 'S': 0}
df['size'] = df['size'].map(size_map)

f.print_df(lines=31, data_frame=df)


# to reverse
inverse_map = {v: k for k, v in size_map.items()}
df['size'] = df['size'].map(inverse_map)

f.print_df(lines=31, data_frame=df)


# class labels
class_map = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
f.print_df(lines=31, data_frame=class_map)

df['classlabel'] = df['classlabel'].map(class_map)

f.print_df(lines=31, data_frame=df)

# per ritornare alle label iniziali

inv_class_map = {k: v for v, k in class_map.items()}
df['classlabel'] = df['classlabel'].map(inv_class_map)

f.print_df(lines=31, data_frame=df)


# PIù SEMPLICEMENTE SCIKIT LO FA IN AUTOMATICO E L'INVERSE è MOLTO PIù SEMPLICE!

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
df['classlabel'] = y

f.print_df(lines=31, data_frame=y)
f.print_df(lines=31, data_frame=df)


# y_inv = class_le.inverse_transform(y)
# df['classlabel'] = y_inv
#
# f.print_df(lines=31, data_frame=y_inv)
# f.print_df(lines=31, data_frame=df)

# encode colors

df['size'] = df['size'].map(size_map)
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

f.print_df(lines=31, data_frame=X)

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
X_ohe = ohe.fit_transform(X)

f.print_df(X_ohe, lines=50)

# oppure la stessa cosa si fa con pandas

X_pd = pd.get_dummies(df[['color', 'size', 'price']])

f.print_df(X_pd, lines=70)


df['size'] = df['size'].map(inverse_map)
X_pd = pd.get_dummies(df[['color', 'size', 'price']])

f.print_df(X_pd, lines=70)
