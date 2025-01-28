from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

students = [
    [85, "M", "good"],
    [90, "F", "excellent"],
    [np.nan, "F", "verygood"],
    [60, "M", "ok"],
    [88, "M", "good"]
]

cols = ["Marks", "Gender", "score"]

df = pd.DataFrame(students, columns=cols)
print(df)

imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

result = df["Marks"].values.reshape(-1, 1)
df.Marks = imputer.fit_transform(result)

print()
print(df)
