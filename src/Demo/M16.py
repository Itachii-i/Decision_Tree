import pandas as pd
from sklearn.linear_model import LinearRegression

df= pd.read_csv("homeprices2.csv")
print(df)
print()

dummies = pd.get_dummies(df.town, dtype= int)

print(dummies)
print()

cities = df.town.unique()

for city in cities:
    print(df[df.town == city])
print()

merged= pd.concat([df, dummies], axis='columns')
print(merged)
print()

final = merged.drop(['town'], axis='columns')

print(final)

X = final.drop('price', axis='columns')
y = final.price

print(X)
print(y)

print()

model_l = LinearRegression()
model_l.fit(X,y)

print("Model trained")
print("prediction")

p = model_l.predict(X)
print(p)
print()
print(model_l.score(X, y))

print("predicting house price in Gudiwada", model_l.predict([[3000, 1, 0, 0]]))
print("predicting house price in Guntur", model_l.predict([[3000, 0, 1, 0]]))
print("predicting house price in Vijayawada", model_l.predict([[3000, 0, 0, 1]]))