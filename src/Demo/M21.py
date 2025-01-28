from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

digits= load_digits()

print(digits)
print("splitting the dataset")

X_train, X_test, y_train, y_test=train_test_split(digits.data, digits.target, test_size=7)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test,y_test))
print(model.predict([digits.data[6]]))


