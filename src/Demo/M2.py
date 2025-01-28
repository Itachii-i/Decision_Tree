import pandas as pd
from scipy import stats

d = {
    "a": [500, 600, 70, 20, 100],
    "b": [55000, 69000, 80000, 10000, 899000]
}
df = pd.DataFrame(d)

X = df.a.values
Y = df.b.values

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

print(r_value)



