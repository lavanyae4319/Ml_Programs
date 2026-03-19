import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.array([10,20,30,40,50]).reshape(-1,1)
y=np.array([12,15,19,31,43])
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred,color='red')
print("slope",model.coef_)
print("intercept:",model.intercept_)
plt.show()
