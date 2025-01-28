import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv("homeprices.csv")
new_df = df.drop("price", axis="columns")

model=LinearRegression()
model.fit(new_df.values,df.price.values)

#print(model.predict([[5000]]))
with open("model_pickle", "wb")as file:
 pickle.dump(model, file)

with open("model_pickle", "rb")as file:
 model1 = pickle.load(file)
 print(model1.predict([[5000]]))

with open("model_pickle", "rb") as file:
 model1= pickle.load(file)
 print(model1.predict([[6000]]))

