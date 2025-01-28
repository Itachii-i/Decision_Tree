from numpy import asarray
from sklearn.preprocessing import PolynomialFeatures

data1 = asarray([[2,3],[4,5],[6,7]])

poly = PolynomialFeatures(degree= 3)
data2 = poly.fit_transform(data1)

print(data1)
print()
print(data2)