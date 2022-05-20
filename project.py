# %%
import pandas as pd 
import numpy as np 
import altair as alt 
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from mlxtend.evaluate import bias_variance_decomp


# Put data in dataframe and quick view it
df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")
print(df_train.head())
print(df_train.columns)

# %%

df_train['date_useful'] = df_train['date'].str[:8] 

df_train['date_useful'] = df_train['date_useful'].astype('|f4')

# Removed ID, View

# sqft_living15, grade, sqft_living, 

features = df_train.filter(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15', 'price', 'date_useful'])

print(df_train.head())

X = features.drop(['price'] , axis=1)
y = df_train.filter(["price"] , axis=1)


X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=.2 , random_state=0 )


vr = VotingRegressor([
                ('gbr', GradientBoostingRegressor(random_state=0).fit(X_train, y_train)), 
               ('lgbm', LGBMRegressor(num_leaves= 3, random_state=0).fit(X_train, y_train))]).fit(X_train, y_train)
               

y_pred = vr.predict(X_test)

print(f"Model Score: {vr.score(X_test, y_test)}")
print(f" Square Root Mean Squared Error: {math.sqrt(mean_squared_error(y_pred, y_test))}")
print(f"R2 score: {r2_score(y_pred, y_test)}")

mse, bias, var = bias_variance_decomp(vr, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)



# %%
