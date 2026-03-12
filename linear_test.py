from sklearn.linear_model import LinearRegression

X = [[1000], [1500], [2000]]  # Square footage

y = [200000, 250000, 300000]  # House prices

model = LinearRegression().fit(X, y)

print(model.predict([[1800]]))  # Predict price for 1800 sq ft