import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("homeprices.csv")

print(df.head())
print()

new_df = df.drop("price", axis="columns")

print(new_df)
print()

print(new_df.values)
print()

print(df.price.values)

reg = LinearRegression()
reg.fit(new_df.values, df.price.values)

print(reg.intercept_)
print(reg.predict([[5000]]))

area_df = pd.read_csv("areas.csv")

p = reg.predict(area_df.values)

area_df['price'] = p
area_df.to_csv("output.csv")
print("check output file")












