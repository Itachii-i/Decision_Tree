import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("insurance_data.csv")
print(df)

X= df[["age"]].values
y=df.bought_insurance

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=7,random_state=0)
print()

print("X_train")
print(X_train)
print()
print("X_test")
print(X_test)
print()

print("y_train")
print(y_train)
print()
print("y_test")
print(y_test)
print()

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained")

print("predict for 50", model.predict([[50]]))
print("Predict for 70", model.predict([[70]]))

y_pred= model.predict(X_test)

print("X_test data is \n")
print(X_test,'\n')

print("y_pred is \n")
print(y_pred, "\n")


