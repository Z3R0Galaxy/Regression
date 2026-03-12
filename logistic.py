import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

hours = np.array([[1],[2],[3],[4],[5],[6]])
pass_exam = np.array([0,0,0,1,1,1])

model = LogisticRegression()
model.fit(hours,pass_exam)

print(model.predict([[1.5]])) #0 is fail and 1 is pass

plt.scatter(hours, pass_exam)

x_range = np.linspace(hours.min(), hours.max(), 100).reshape(-1,1)

probabilities = model.predict_proba(x_range)[:,1]

plt.plot(x_range, probabilities)

plt.title("Logistic Regression Probability Curve")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")

plt.show()