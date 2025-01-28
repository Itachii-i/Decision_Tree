import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df= pd.read_csv("poly_dataset.csv")
print(df)
print()
X= df.iloc[:, 1:2].values
print(X)

y=df.iloc[:, 2].values
print(y)

poly_r= PolynomialFeatures(degree= 5)
x_poly= poly_r.fit_transform(X)

model=LinearRegression()
model.fit(x_poly,y)

#plotting
plt.scatter(X,y, color="red")
plt.plot(X,model.predict(x_poly), color="blue")

plt.show()

