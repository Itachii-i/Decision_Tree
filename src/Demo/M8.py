import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Salary_Data.csv")

X=df.iloc[:, :-1].values
Y=df.iloc[:,1].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)

print("X_train:", X_train)
print()
print("X_test:", X_test)
print()
print("Y_train:", Y_train)
print()
print("Y_test:", Y_test)
print()

reg=LinearRegression()
reg.fit(X_train, Y_train)

print("Model trained")

Y_pred= reg.predict(X_test)
print(Y_pred)
print()

d = {"Actual":Y_test, "Predicted": Y_pred}
compare_df =pd.DataFrame(d)
print(compare_df)

