from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])
y = np.array([3.2, 4.1, 6.0, 9.3, 13.8, 19.7, 26.9, 35.8, 45.9, 58.2, 71.8, 87.9])

poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly,y)

prediction = model.predict(poly.transform([[13]]))
print(prediction)

plt.scatter(x,y)
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1,1)

x_range_poly = poly.transform(x_range)

plt.plot(x_range, model.predict(x_range_poly))

plt.title("Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()