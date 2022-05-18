# %%
import pandas as pd 
import numpy as np 
import altair as alt 
import seaborn as sns

# Put data in dataframe and quick view it
df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")
print(df_train.head())
print(df_train.columns)

# %%
features = df_train.filter(['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15', 'price'])


# %%
'''
Function to show bar graph about the passed in feature
'''
def show_bar_graph(feature):
    chart = alt.Chart(df_train).mark_bar().encode(
        alt.X(f"{feature}:T", bin=True),
        y="price"
    )
    return chart

yr_built = show_bar_graph("yr_built")
condition = show_bar_graph("condition")
grade = show_bar_graph("grade")
sqft_living = show_bar_graph("sqft_living")
sqft_lot15 = show_bar_graph("sqft_lot15")
sqft_lot15



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
)



# %%
# See which features correlate to the price the most
corr = features.corr()
sns.heatmap(corr)
#==================================================================================================================
# sqrft_living, grade, sqft_above, bathrooms - features that correlate most to the price
# zipcode, long, condition, sqft_lot - features that correlate least to the price. 
#==================================================================================================================
sns.pairplot(features, hue = 'price')
# %%
