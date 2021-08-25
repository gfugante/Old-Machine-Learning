
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
from io import StringIO


csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,6.0
    0.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)

print()
print(23*'-')
print()

# df = df.dropna()

print(df.dropna(thresh=4))

print()
print(23*'-')
print()

imr = Imputer(strategy='most_frequent', axis=1) # prende il più fequente sulle y, altrimenti il primo
imr.fit(df)
imputed_data = imr.transform(df.values)

print(imputed_data)

print()
print(23*'-')
print()



