import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

url="https://raw.githubusercontent.com/digipodium/Datasets/main/regression/dataA.csv"
df = pd.read_csv(url)

#%%
from sklearn.model_selection import train_test_split
X, y = df.drop(columns=['y', 'z']), df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)

#%%
Xp2 = PolynomialFeatures(degree=2).fit_transform(X)

#%%
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)*100
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#%%

