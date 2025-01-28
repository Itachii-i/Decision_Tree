from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits= load_digits()
print(len(digits.data))
print()
print(dir(digits))
print()
print(digits.DESCR)
print()
print(digits.data)
print()
print(digits.data[0])
print()

for i in range(10):
    plt.matshow(digits.images[i])
    plt.pink()
    plt.show()