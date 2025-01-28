import pandas as pd
from sklearn.linear_model import LinearRegression

df= pd.read_csv("homeprices1.csv")

print(df)

#Mean of the bedrooms

m= df.bedrooms.median()

df.bedrooms = df.bedrooms.fillna(m)
print(df)
print()

a= df.drop('price', axis="columns")
print(a)

leg= LinearRegression()
leg.fit(a.values, df.price)

print("Intercept :", leg.intercept_)
print("Coeff:", leg.coef_)

print(leg.predict([[3000,3,40]]))

print()

print(leg.predict([[2500,4, 5]]))

print(leg.predict([[5000, 7, 3]]))