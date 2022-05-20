#Function to show bar graph about the passed in feature


# %%
import pandas as pd 
import numpy as np 
import altair as alt 
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor

df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")

def show_bar_graph(feature):
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(df_train).mark_bar().encode(
        alt.X(f"{feature}:T", bin=True),
        y="mean(price)"
    )
    chart.save(f"{feature}.png")



show_bar_graph("condition")


show_bar_graph("grade")
#%%
# print(df_train["grade"].value_counts())



show_bar_graph("sqft_living")
show_bar_graph("sqft_lot15")



# %%
alt.Chart(df_train).mark_circle().encode(
    alt.X("bedrooms", bin=True),
    y='price',
)

# %%
alt.data_transformers.disable_max_rows()

alt.Chart(df_train).mark_circle().encode(
    alt.X("bedrooms"),
    y='price',
    bin=True
)
'''

'''

# %%
# See which features correlate to the price the most
#corr = features.corr()
#sns.heatmap(corr)
#==================================================================================================================
# sqrft_living, grade, sqft_above, bathrooms - features that correlate most to the price
# zipcode, long, condition, sqft_lot - features that correlate least to the price. 
#==================================================================================================================
#sns.pairplot(features, hue = 'price')
# %%