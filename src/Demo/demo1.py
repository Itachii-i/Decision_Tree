print("Step 1: Importing libraries")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

print("Step 2: Loading the dataset")

df = pd.read_csv('student_scores.csv')



print("Step 3: Creating feature and target")

X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

print(X,y)



print("Step 4: Splitting the data from feature and target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("Step 5: Model creation")

model = LinearRegression()


print("Step 6: Model training")

model.fit(X_train, y_train)


print("Step 7: Prediction for X_test values")


y_pred = model.predict(X_test)


print("Step 8: Regression metrics")

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print()

print("mae", mae)
print("mse",mse)
print("rmse",rmse)
