import pandas as pd
from sklearn.linear_model import LinearRegression

df= pd.read_csv("homeprices1.csv")
print(df)

print("Median of the bedrooms")
print(df.bedrooms.median())

m = df.bedrooms.median()
df.bedrooms = df.bedrooms.fillna(m)
print(df)

a = df.drop('price', axis ='columns')

print("Model training")
L_model= LinearRegression()
L_model.fit(a.values, df.price)

print("Intercept:",L_model.intercept_)
print("coeffecitent:", L_model.coef_)

print("Prediction")

print(L_model.predict([[3000,3,40]]))

