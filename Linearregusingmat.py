import numpy as np
import matplotlib.pyplot as plt

x= np.array([1,2,3,4,5])
y= np.array([2,5,7,10,11])

x_mean= np.mean(x)
y_mean= np.mean(y)

numerator=np.sum((x-x_mean)*(y-y_mean))
np.sum((x-x_mean)*(y-y_mean))
denominator= np.sum((x-x_mean)**2)

w= numerator/denominator
b= y_mean - w*x_mean

y_pred= w*x + b

print("Slope (w):", w)
print("Intercept (b):", b)

plt.scatter(x,y)
plt.plot(x,y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Matrices')
plt.show()


y_pred = w*x[0] + b
loss = np.sum((y[0] - y_pred)**2)/len(x)
print("Loss:", loss)
