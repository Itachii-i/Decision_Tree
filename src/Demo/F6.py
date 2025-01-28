from sklearn.preprocessing import OneHotEncoder
from numpy import asarray

d = asarray([["green"],["blue"],["red"],["aqua"]])

encoder = OneHotEncoder(drop="first", sparse_output= False)
result = encoder.fit_transform(d)
print(d)
print(result)
