import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

def linear_regression(x, y):
  model = LinearRegression().fit(x, y)

  coefficient_of_determination = model.score(x, y) # R^2
  slope = model.coef_
  interception = model.intercept_

  return (coefficient_of_determination, slope, interception)

def linear_regression_output(q_sq_2, slope, interception, prediction_function):
  coefficient_of_determination = q_sq_2

  print('The linear regression model is y = %.2f*x + %.2f' % (slope, interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction_function(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )



def quadratic_regression(x, y):
  transformer = PolynomialFeatures(degree = 2, include_bias = False)
  transformer.fit(x)
  x_ = transformer.transform(x)

  model = LinearRegression().fit(x_, y)
  
  coefficient_of_determination = model.score(x_, y)
  interception = model.intercept_
  coefficients = model.coef_

  return (coefficient_of_determination, coefficients, interception)
  
def quadratic_regression_output(q_sq_2, coefficients, interception, prediction_function):
  coefficient_of_determination = q_sq_2

  print('The quadratic regression model is y = %.4f*x^2 + %.4f*x + %.4f' % (coefficients[0], coefficients[1], interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )


# Linear regression
coefficient_of_determination, slope, interception = linear_regression(x, y)
prediction = lambda age: slope * age + interception

linear_regression_output(coefficient_of_determination, slope, interception, prediction)


# Quadratic regression
coefficient_of_determination, coefficients, interception = quadratic_regression(x, y)
prediction = lambda age: coefficients[-1] * age**2 + coefficients[-2]*age + interception

quadratic_regression_output(coefficient_of_determination, coefficients, interception, prediction)
