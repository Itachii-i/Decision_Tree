import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("poly_dataset.csv")
print(df)

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

print(X)
print(y)
#plt.scatter(X, y, color="red")
#plt.xlabel("Position")
#plt.ylabel("salary")

#plt.show()

model_p = PolynomialFeatures(degree=4)
x_poly = model_p.fit_transform(X)

model= LinearRegression()
model.fit(x_poly, y)
print("fitting the polynomial regression to the dataset")

#plotting Polynomial Regression

#plt.scatter(X, y, color="blue")
#plt.plot(X, model.predict(x_poly), color="red")

#plt.show()

print("Prediction")
pred_p = model.predict(model_p.fit_transform([[6.5]]))
print(pred_p)

