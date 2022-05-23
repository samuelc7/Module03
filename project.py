# %%
from pyexpat import model
from re import sub
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
import itertools
import threading

class Model:
    def __init__(self, features, df_train):
        self.features = features
        self.df_train = df_train.filter(features)
        self.model_score = None
        self.sqrt_mean_squared_error = None
        self.r2_score = 0
        self.vr = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_tes = None
        self.y_pred = None

    def setup(self):
        if "date" in self.features:
            self.df_train['date_useful'] = self.df_train['date'].str[:8] 
            self.df_train['date_useful'] = self.df_train['date_useful'].astype('|f4')

        X = self.df_train.drop(['price'] , axis=1)
        y = self.df_train.filter(["price"] , axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                X, y, test_size=.2 , random_state=0 )

        self.vr = VotingRegressor([
                ('gbr', GradientBoostingRegressor(random_state=0).fit(self.X_train, self.y_train))]).fit(self.X_train, self.y_train)#, 
               #('lgbm', LGBMRegressor(num_leaves= 3, random_state=0).fit(self.X_train, self.y_train))]).fit(self.X_train, self.y_train)

        self.y_pred = self.vr.predict(self.X_test)
        self.model_score = self.vr.score(self.X_test, self.y_test)
        self.sqrt_mean_squared_error = math.sqrt(mean_squared_error(self.y_pred, self.y_test))
        self.r2_score = r2_score(self.y_pred, self.y_test)

    def display(self, to_file=False):
        
        if to_file: 
            out_put = f'''
                    {"*" * 120}\n
                    Features: {self.features}\n
                    {"-" * 60}\n
                    Model Score: {self.model_score}\n
                    Square Root Mean Squared Error: {self.sqrt_mean_squared_error}\n
                    R2 score: {self.r2_score}\n
                    {"-" * 60}\n
                    {"*" * 120}
                '''

            return out_put
        
        print("#" * 60)
        print(f"Features: {self.features}")
        print("-" * 60)

        print(f"Model Score: {self.model_score}")
        print(f"Square Root Mean Squared Error: {self.sqrt_mean_squared_error}")
        print(f"R2 score: {self.r2_score}")
        print("-" * 60)
        print("#" * 60)

#%%
df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")

df_train['avg_zipPrice'] = df_train['price'].groupby(df_train['zipcode']).transform('mean')


features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
        'sqft_living15', 'sqft_lot15', 'price', 'date_useful', 'avg_zipPrice']

sns.scatterplot(data=df_train, x="zipcode", y="avg_zipPrice")


#%%

model1 = Model(features, df_train)
model1.setup()
model1.display()
# %%

# # 6 to keep at all times: sqft_living15, grade, sqft_living, bathrooms, sqft_above, price
base_features = ["sqft_living15", "grade", "sqft_living", "bathrooms", "sqft_above", "price"]
other_features = ['bedrooms', 'sqft_lot', 'floors', 'waterfront', 'condition',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
        'sqft_lot15', 'date_useful']


# for i in other_features: 
#     base_features.append(i)
#     model = Model(base_features, df_train)
#     model.setup()
#     model.display()


# for L in range(0, len(features)+1):
#     for subset in itertools.combinations(features, L):
#         subset = list(subset)
#         if len(subset) > 8:
#             if "price" not in subset:
#                 subset.append("price")
#             model = Model(subset, df_train)
#             model.setup()
#             print(model.r2_score)
#             if model.r2_score > .75:
#                 with open("out.txt", "a") as f:
#                     f.write(f"{model.display(to_file=True)}\n")

data = {
    'predictions' : model1.y_pred , 'actual' : model1.y_test['price']
}

df_factor = df = pd.DataFrame(data)

print(df_factor.shape)

alt.Chart(df_factor).mark_point().encode(
    alt.X('actual', scale=alt.Scale(domain=[0, 3000000])),
    y= 'predictions',
    color='predictions' 
)
                
# %%
