from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
print(iris)
print()
print(dir(iris))
print()
print(iris.feature_names)
print()
print(iris.target_names)

df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
df["target"]= iris.target

print(df[df.target ==0].head())
print(df[df.target==1].head())
print(df[df.target==2].head())

a=lambda x:iris.target_names[x]
df['flower_name']= df.target.apply(a)
print(df)

setosa_50 = df[:50]
print(setosa_50.head())


versicolor_50 = df[50:100]
print(versicolor_50.head())

verginica_50= df[100:]
print(verginica_50.head())

X= df.drop(["target", "flower_name"], axis= "columns")
y=df.target

X_train, X_test,y_train,y_test= train_test_split(X,y, test_size=0.2)

print("splitting the data")

RC = RandomForestClassifier(n_estimators=40)
RC.fit(X_train.values,y_train)

print("model trained")

print(RC.predict([[6.7, 3.0, 5.2, 2.3]]))
print()
print(RC.score(X_test,y_test))
