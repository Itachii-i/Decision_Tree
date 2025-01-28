import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("salaries.csv")
print(df)

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

print(inputs.head())
print(target)

lb= LabelEncoder()

inputs['company_n'] = lb.fit_transform(inputs['company'])
print(inputs)

inputs['job_n']= lb.fit_transform(inputs['job'])
inputs['degree_n']= lb.fit_transform(inputs['degree'])

print(inputs)

inputs_n= inputs.drop(['company','job','degree'], axis="columns")

print(inputs_n)

model= DecisionTreeClassifier()

print("model creation")

model.fit(inputs_n.values, target)

print("model trained")

print(model.score(inputs_n.values, target))

print(model.predict([[2,2,1]]))

