from numpy import asarray
from sklearn.preprocessing import PolynomialFeatures

d = asarray([[2,3],[4,5],[6,7]])

poly = PolynomialFeatures(degree= 2)

d1 = poly.fit_transform(d)

print(d1)