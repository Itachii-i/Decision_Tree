import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv("online_shoppers_intention.csv")
print(df.head())
print(df.shape)
print(df.info())

print(df["Weekend"].unique())

# converting boolean to int
df["Weekend"]= df["Weekend"].astype(int)

df["Revenue"]= df["Revenue"].astype(int)
print(df.head())

print(df["VisitorType"].unique())

# adding Returning visitor column to df

condition = df["VisitorType"]=="Returning_Visitor"
df["Returning_Visitor"]= np.where(condition,1,0)
print(df.head())

df= df.drop(columns=["VisitorType"])
print(df.head())

print(df.Month.unique())

OD= OrdinalEncoder()

df["Month"] = OD.fit_transform(df[["Month"]])
print(df.Month.unique())

print(df.Revenue.value_counts())

result=df[df.columns[1:]].corr()["Revenue"]
print(result)
result1= result.sort_values(ascending=False)
print(result1)

#Prepare X,y values

X=df.drop(["Revenue"],axis=1)
y= df["Revenue"]

print("Features and Targets are created")

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=0)
print("Train and test datasets are created")
print()

model = SVC()
model.fit(X_train, y_train)
print("Model training by SVC")

y_pred = model.predict(X_test)

#Evaluating the model
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test,y_test))
print()

print("Confusion Metrics")
cr= metrics.classification_report(y_test,y_pred)
print(cr)
print()

print("Naive Bayes")
model= GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training Accuracy:", model.score(X_train,y_train))
print("Testing Accuracy", model.score(X_test,y_test))

