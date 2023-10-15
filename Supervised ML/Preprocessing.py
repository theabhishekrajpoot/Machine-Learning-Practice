# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:45:07 2023

@author: Abhis
"""
#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
df = pd.read_csv('https://raw.githubusercontent.com/digipodium/Datasets/main/sample_data.csv')
df

#%%
si=SimpleImputer()
df[['salary', 'age']]=si.fit_transform(df[['salary', 'age']])
df

#%%
minmax=MinMaxScaler()
minmax.fit_transform(df[['salary', 'age']])

#%%
stdsclr=StandardScaler()
stdsclr.fit_transform(df[['salary', 'age']])

#%%
from sklearn.preprocessing import OrdinalEncoder
ordEnc=OrdinalEncoder()
df[['country', 'happy']]=ordEnc.fit_transform(df[['country', 'happy']])
df

