import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

transformer = PolynomialFeatures(degree = 2, include_bias = False)
transformer.fit(x)
x_ = transformer.transform(x)

model = LinearRegression().fit(x_, y)

coefficient_of_determination = model.score(x_, y)
interception = model.intercept_
coefficients = model.coef_

prediction = lambda age: coefficients[0] * age**2 + coefficients[1]*age + interception


print('The quadratic regression model is y = %.4f*x^2 + %.4f*x + %.4f' % (coefficients[0], coefficients[1], interception))

age = int(input('what is your age: '))
print('your recomended HR is %d' % (prediction(age)))
print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )
# model = LinearRegression().fit(x, y)

# coefficient_of_determination = model.score(x, y) # R^2
# slope = model.coef_
# interception = model.intercept_


# print('The linear regression model is y = %.2f*x + %.2f' % (slope, interception))

# age = int(input('what is your age: '))
# print('your recomended HR is %d' % (prediction(age)))
# print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )