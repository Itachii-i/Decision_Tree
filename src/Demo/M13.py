
#Dummy variable
print("Importing Libraries")
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split

print("Loading the dataset")
df= pd.read_csv("homeprices2.csv")
print(df)

dummies = pd.get_dummies(df.town, dtype= int)
merged = pd.concat([df,dummies], axis="columns")

print(merged)
print()

final= merged.drop(["town"], axis="columns")

print(final)

X =final.drop("price", axis="columns")
y = final.price
print()
print(X.values)
print()
print(y.values)

model= LinearRegression()
model.fit(X.values,y.values)

print("Model trained")

print(model.predict(X.values))

print(model.score(X.values,y.values))

print(model.predict([[3600, 0,1, 0]]))
print(model.predict([[3400, 0,0, 1]]))


