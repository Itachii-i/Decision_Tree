from sklearn.preprocessing import LabelEncoder
import pandas as pd

d = {"Company": ["MS", "Google", "MS", "Apple"],
     "Role": ["HR", "Agent", "HR", "SD"]}
df = pd.DataFrame(d)

encoder = LabelEncoder()

df["comapany_n"] = encoder.fit_transform(df["Company"])
df["role_n"] = encoder.fit_transform(df["Role"])


print(df)
