from sklearn.preprocessing import OrdinalEncoder
from numpy import asarray

d = asarray([["green"],["blue"],["red"],["aqua"]])

encoder = OrdinalEncoder()
result = encoder. fit_transform(d)

print(d)
print(result)

