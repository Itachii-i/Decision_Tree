import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df= pd.read_csv("student_scores.csv")

X=df.iloc[:,:-1].values
Y= df.iloc[:,1].values

print(X)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

print( )
print("X_train:", X_train)
print()
print("X_test:", X_test)
print()
print("Y_train: ", Y_train)
print()
print("Y_test: ", Y_test)
print()

reg=LinearRegression()
reg.fit(X_train, Y_train)
print("Modal trained")

print(reg.coef_)
print()
print(reg.intercept_)
print()

Y_pred= reg.predict(X_test)
print("Y prediction:",Y_pred)
print()

d={"Actual":Y_test, "Predicted": Y_pred}

compare_df= pd.DataFrame(d)

print(compare_df)
print()

mae = metrics.mean_absolute_error(Y_test, Y_pred)
mse = metrics.mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))

print("Mean AE:", mae)
print("Mean SQ Error:", mse)
print("Root mean SQ error:", rmse)
