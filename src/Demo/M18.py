import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("Salary_Data.csv")

print(df)

X=df.iloc[:,0:1].values
y=df.iloc[:,-1].values

print(X)
print(y)

X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=1/3, random_state=0)

print()
print(X_train)
print()
print(X_test)
print()
print(y_train)
print()
print(y_test)

model=LinearRegression()
model.fit(X_train, y_train)

y_pred= model.predict(X_test)

print("Predicting the salaries")
print(y_pred)
print()
print(y_test)
