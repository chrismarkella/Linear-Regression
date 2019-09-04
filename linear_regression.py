import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

def linear_regression(x, y):
  """
  Return the coefficient of determination (R^2), slope and interception
  """
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

  print('The quadratic regression model is y = %.4f*x^2 + %.4f*x + %.4f' % (coefficients[-1], coefficients[-2], interception))
  # print('The quadratic regression model is y = %.4f*x + %.4f*x^2 + %.4f' % (coefficients[0] , coefficients[1] , interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )

def polynomial_regression(x, y, degree_my = 1):
  transformer = PolynomialFeatures(degree = degree_my, include_bias = False)
  transformer.fit(x)
  x_ = transformer.transform(x)

  model = LinearRegression().fit(x_, y)
  
  coefficient_of_determination = model.score(x_, y)
  interception = model.intercept_
  coefficients = model.coef_

  return (coefficient_of_determination, coefficients, interception)

def cubic_regression(x, y):
  transformer = PolynomialFeatures(degree = 3, include_bias = False)
  transformer.fit(x)
  x_ = transformer.transform(x)

  model = LinearRegression().fit(x_, y)
  
  coefficient_of_determination = model.score(x_, y)
  interception = model.intercept_
  coefficients = model.coef_

  return (coefficient_of_determination, coefficients, interception)

def cubic_regression_output(q_sq_2, coefficients, interception, prediction_function):
  coefficient_of_determination = q_sq_2

  print('The cubic regression model is y = %.4f*x^3 + %.4f*x^2 + %.4f*x + %.4f' % (coefficients[0], coefficients[1], coefficients[2], interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )


def polynomial_regression_output(q_sq_2, coefficients, interception, prediction_function):
  degree = len(coefficients)
  coefficient_of_determination = q_sq_2

  print('The polynomial regression model with degree %d is' %(degree) )
  model_txt = ' + '.join([str(coefficients[i]) + '*x^' + str(i+1) for i in range(len(coefficients))])
  model_txt = model_txt + ' + ' + str(interception)
  print(model_txt)

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )


# # Linear regression
# coefficient_of_determination, slope, interception = linear_regression(x, y)
# prediction = lambda age: slope * age + interception

# linear_regression_output(coefficient_of_determination, slope, interception, prediction)


# # Quadratic regression
# coefficient_of_determination, coefficients, interception = quadratic_regression(x, y)
# prediction = lambda age: coefficients[-1] * age**2 + coefficients[-2]*age + interception
# prediction2 = lambda age: coefficients[0] * age + coefficients[1]*age**2 + interception
# prediction3 = lambda age: sum([coefficients[i]*age**(i + 1) for i in range(len(coefficients))]) + interception

# quadratic_regression_output(coefficient_of_determination, coefficients, interception, prediction)
# quadratic_regression_output(coefficient_of_determination, coefficients, interception, prediction2)
# quadratic_regression_output(coefficient_of_determination, coefficients, interception, prediction3)

# # Cubic regression
# coefficient_of_determination, coefficients, interception = cubic_regression(x, y)
# prediction = lambda age: coefficients[-1] * age**3 + coefficients[-2]*age**2 + coefficients[-3]*age+ interception

# cubic_regression_output(coefficient_of_determination, coefficients, interception, prediction)

# # polynomial regression with degree 3
# coefficient_of_determination, coefficients, interception = polynomial_regression(x, y, 3)
# print(coefficient_of_determination, coefficients, interception)

# prediction = lambda age: sum([coefficients[i]*age**(i + 1) for i in range(len(coefficients))]) + interception
# # prediction = lambda age: coefficients[-1] * age**3 + coefficients[-2]*age**2 + coefficients[-3]*age+ interception

# polynomial_regression_output(coefficient_of_determination, coefficients, interception, prediction)

# # polynomial regression with degree 5
# coefficient_of_determination, coefficients, interception = polynomial_regression(x, y, 5)
# print(coefficient_of_determination, coefficients, interception)

# prediction = lambda age: sum([coefficients[i]*age**(i + 1) for i in range(len(coefficients))]) + interception
# # prediction = lambda age: coefficients[-1] * age**3 + coefficients[-2]*age**2 + coefficients[-3]*age+ interception

# polynomial_regression_output(coefficient_of_determination, coefficients, interception, prediction)

best_three_polynomial_regression = lambda x,y, max_degree: sorted([(polynomial_regression(x,y,i)[0], i) for i in range(2, max_degree)], reverse = True)[:3]

print('the top three polynomial regressions are')
for regression in best_three_polynomial_regression(x, y, 150):
  print('%.3f Q^2 with degree %d' %(regression[0], regression[1]))