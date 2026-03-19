import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)
df = df[['horsepower','mpg']]
df = df.dropna()


x= df["horsepower"].values.reshape(-1,1)

y= df["mpg"]

model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred,color='red')
print("slope",model.coef_)
print("intercept:",model.intercept_)
plt.show()

