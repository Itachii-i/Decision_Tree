import pandas as pd
from sklearn.linear_model import LinearRegression

df= pd.read_csv("homeprices1.csv")
print(df)

print("Median:", df.bedrooms.median())

m = df.bedrooms.median()

df.bedrooms = df.bedrooms.fillna(m)
print()
print(df)

a= df.drop('price', axis='columns')

model = LinearRegression()
model.fit(a.values, df.price)

print('model trained')
print()

print("Intercept:", model.intercept_)
print()

print("Coeff:", model.coef_)

print("price for 300sft, 3bedrooms,40 years:", model.predict([[300, 3, 40]]))

b=  112.06244194 * 300 + 23388.88007794 * 3 + -3231.71790863 * 40

print(b)