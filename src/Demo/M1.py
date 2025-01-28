from sklearn.model_selection import train_test_split
import numpy as np

d= np.arange(10)

X_train,X_test = train_test_split(d)

print(d)
print()
print(X_train)
print(X_test)