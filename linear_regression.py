
# Used this tutorial: https://realpython.com/linear-regression-in-python/

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Got the datapoins from
# https://johnstonmd.wordpress.com/teaching/math-178-spring-2017/
x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

def polynomial_regression(x, y, degree_my = 1):
  transformer = PolynomialFeatures(degree = degree_my, include_bias = False)
  transformer.fit(x)
  x_ = transformer.transform(x)

  model = LinearRegression().fit(x_, y)
  
  coefficient_of_determination = model.score(x_, y)
  interception = model.intercept_
  coefficients = model.coef_

  return (coefficient_of_determination, coefficients, interception)

def linear_regression(x, y):
  """
  Returns R^2, slope and interception
  """
  # Reusing the existing 'general' polynomial_regression function
  return polynomial_regression(x, y, 1) 
  

def linear_regression_output(q_sq_2, slope, interception, prediction_function):
  coefficient_of_determination = q_sq_2

  print('The linear regression model is y = %.2f*x + %.2f' % (slope, interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction_function(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )

def quadratic_regression(x, y):
  """
  Returning R^2, coefficients and the interceptions as a size three tupple.
  """
  return polynomial_regression(x, y, 2)

def quadratic_regression_output(q_sq_2, coefficients, interception, prediction_function):
  coefficient_of_determination = q_sq_2

  print('The quadratic regression model is y = %.4f*x^2 + %.4f*x + %.4f' % (coefficients[-1], coefficients[-2], interception))
  # print('The quadratic regression model is y = %.4f*x + %.4f*x^2 + %.4f' % (coefficients[0] , coefficients[1] , interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction_function(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )


def cubic_regression(x, y):
  """
  Returning R^2, coefficients and the interceptions as a size three tupple.
  """
  return polynomial_regression(x, y, 3)

def cubic_regression_output(q_sq_2, coefficients, interception, prediction_function):
  coefficient_of_determination = q_sq_2

  print('The cubic regression model is y = %.4f*x^3 + %.4f*x^2 + %.4f*x + %.4f' % (coefficients[0], coefficients[1], coefficients[2], interception))

  age = int(input('what is your age: '))
  print('your recomended HR is %d' % (prediction_function(age)))
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


def top_polynomial_regressions(x, y, max_degree, top = 10):
  """
  Print out the best polynomial regressions with the highest Q^2 in a sorted list 
  """
  sorted_polynomial_regressions = lambda x,y, max_degree: sorted([(polynomial_regression(x,y,i)[0], i) for i in range(2, max_degree)], reverse = True)

  print('the top %d polynomial regressions are' %top)
  for regression in sorted_polynomial_regressions(x, y, max_degree)[:top]:
    print('%.4f Q^2 with degree %d' %(regression[0], regression[1]))




if __name__ == "__main__":

  #  '\' is just for to be able to break the line and tabulate nice
  functions = \
  [
    (linear_regression,    linear_regression_output   ),
    (quadratic_regression, quadratic_regression_output),
    (cubic_regression,     cubic_regression_output    )
  ]

  for f_in, f_out in functions:
    r_sq, coefficients, interception = f_in(x,y)
    prediction = \
      lambda age: sum( [coefficients[i] * age**(i + 1) for i in range(len(coefficients))] ) + interception
  
    f_out(r_sq, coefficients, interception, prediction)

  top_polynomial_regressions(x, y, 150, 5)
