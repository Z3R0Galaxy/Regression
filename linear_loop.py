from sklearn.linear_model import LinearRegression

x = [[100], [200], [300]]
y = [50, 100, 150]


model = LinearRegression()
model.fit(x,y)

while True:
    num = int(input("Enter number: "))
    output = model.predict([[num]])
    print(output[0])