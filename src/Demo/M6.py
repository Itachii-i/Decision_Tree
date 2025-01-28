import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('student_scores.csv')
print(df)

#df.plot(x= 'Hours', y="Scores", style='o')
#plt.xlabel("Hours studied")
#plt.ylabel('Scores')
#plt.show()

X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

reg= LinearRegression()
reg.fit(X_train, Y_train)

print(reg.intercept_)
print(reg.coef_)

Y_pred = reg.predict(X_test)

d = {'Actual': Y_test, 'Predicted':Y_pred}

compare_df = pd.DataFrame(d)

print(compare_df)
