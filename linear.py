import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[1], [2], [6], [8], [10], [12], [14], [16]])
y = np.array([1, 3, 5, 7, 9, 11, 13, 15])

model = LinearRegression()
model.fit(x, y)

print(model.predict([[68]])) 
print(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

plt.scatter(x,y)
plt.plot(x,model.predict(x))
plt.title("Linear Regression")
plt.show()