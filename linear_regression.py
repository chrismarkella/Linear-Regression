import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

model = LinearRegression().fit(x, y)

coefficient_of_determination = model.score(x, y) # R^2
slope = model.coef_
interception = model.intercept_

prediction = lambda age: slope * age + interception

print('The linear regression model is y = %.2f*x + %.2f' % (slope, interception))

age = int(input('what is your age: '))
print('your recomended HR is %d' % (prediction(age)))
