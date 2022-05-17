
import pandas as pd 
import numpy as np 
import altair as alt 
import seaborn as sns
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")

print(df_train.head())
print(df_train.columns)

features = df_train.filter(['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15', 'price'])

chart1 = alt.Chart(df_train).mark_bar().encode(
    alt.X("yr_built:T", bin=True),
    y='price',
)

chart1.show()

alt.Chart(df_train).mark_circle().encode(
    alt.X("bedrooms", bin=True),
    y='price',
)

alt.data_transformers.disable_max_rows()

alt.Chart(df_train).mark_circle().encode(
    alt.X("bedrooms"),
    y='price',
)

alt.save("chart1.png")



corr = features.corr()
sns.heatmap(corr)
sns.pairplot(features, hue = 'price')