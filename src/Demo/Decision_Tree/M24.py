import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df= pd.read_csv("salaries.csv")
print(df)

input= df.drop("salary_more_then_100k",axis= "columns")
target= df["salary_more_then_100k"]

print("input")
print(input.head())

print("target")
print(target.head())

LE= LabelEncoder()
input["company_n"]= LE.fit_transform(input["company"])
input["job_n"]= LE.fit_transform(input["job"])
input["degree_n"]= LE.fit_transform(input["degree"])

print(input)

input_n = input.drop(["company","job","degree"],axis="columns")

print(input_n)

DT= DecisionTreeClassifier()

print("Decision Tree Classifier object created")



