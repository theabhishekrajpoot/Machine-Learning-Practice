# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:21:15 2023

@author: Abhis
"""
#%%
import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/digipodium/Datasets/main/regression/automobile.csv")

#%%
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


#%%
X = df.drop(columns=['price'])
Y = df['price']

#%%
X.replace('?', np.nan, inplace=True)

#%%
weired_cols = ['normalized-losses', 'stroke', 'horsepower', 'peak-rpm', 'bore']
for col in weired_cols:
    X[col] = X[col].astype(float)
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

#%%
cat_ord_cols=[]
cat_hot_cols=[]
for col in cat_cols:
    print(col, X[col].nunique())
    if X[col].nunique()>2:
        cat_hot_cols.append(col)
    else:
        cat_ord_cols.append(col)
    
#%%
num_pipe = Pipeline(steps=[
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
])

cat_ord_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder()),
])

cat_hot_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first')),
])

transformer = ColumnTransformer(transformers=[
    ('numerical', num_pipe, num_cols),
    ('categorical_hot', cat_hot_pipe, cat_hot_cols),
    ('categorical_ord', cat_ord_pipe, cat_ord_cols),

])

Xpro = transformer.fit_transform(X)












