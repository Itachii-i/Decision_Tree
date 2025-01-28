from sklearn.preprocessing import MinMaxScaler
import pandas as pd

d = {"x": [10, 20, 30, 40, 50],
     "y": [25, 50, 75, 100, 125]
}

df = pd.DataFrame(d)
scale = MinMaxScaler(feature_range=(0, 1))
df[["x", "y"]] = scale.fit_transform(df[["x", "y"]])

print(df)


