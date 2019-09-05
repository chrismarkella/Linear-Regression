
# Used this tutorial: https://realpython.com/linear-regression-in-python/
import sys
import numpy as np
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Got the datapoints from
# https://johnstonmd.wordpress.com/teaching/math-178-spring-2017/
x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

def polynomial_regression(x, y, degree_my = 1):
  """
  Return R^2, coefficients and the interception
  
  polynomial_regression(x, y, degree_my = 1) -> (r_sq, coefficients, interception)

  Parameters
  ----------
  x: np.array, regressors
  y: np.array, predictors
  degree: int, optional, default 1, the degree of the model function
  
  Examples
  --------
  >>> x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
  >>> y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])

  >>> x
  array([[16],
         [17],
         [23],
         [25],
         [28],
         [31],
         [40],
         [48],
         [54],
         [67]])
  >>> y
  array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])
  >>> polynomial_regression(x, y)
  (0.9141779810921304, array([-0.88693692]), 187.95409848808737)
  >>> polynomial_regression(x, y, 2)
  (0.9147602910606555, array([-1.01045344,  0.00153909]), 189.99421074646295)
  >>> polynomial_regression(x, y, 5)
  (0.9184131002053144, array([ 7.73620330e+00, -4.45581354e-01,  1.08361046e-02, -1.26584517e-04,
          5.75609467e-07]), 126.4417938766303)
  >>> 
  """
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

  age = 20
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

  age = 20
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

  age = 20
  print('your recomended HR is %d' % (prediction_function(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )


def polynomial_regression_output(q_sq_2, coefficients, interception, prediction_function):
  degree = len(coefficients)
  coefficient_of_determination = q_sq_2

  print('The polynomial regression model with degree %d is' %(degree) )
  model_txt = ' + '.join([str(round(coefficients[i], 2)) + '*x^' + str(i+1) for i in range(len(coefficients))])
  model_txt = model_txt + ' + ' + str(interception)
  print(model_txt)

  age = 20
  print('your recomended HR is %d' % (prediction_function(age)))
  print('this is %.2f percent accurate' %( coefficient_of_determination * 100) )


def top_polynomial_regressions(x, y, max_degree, top = 10):
  """
  Print out the best polynomial regressions with the highest Q^2 in a sorted list 
  
  Parameters
  ----------
  x: np.array, regressors
  y: np.array, predictors
  max_degree: int, the degree of the model function
  top: int, number of rows printed from the sorted regressions
  

  Examples
  --------
  >>> x = np.array([16, 17, 23, 25, 28, 31, 40, 48, 54, 67]).reshape((-1, 1))
  >>> y = np.array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])
  >>> x
  array([[16],
         [17],
         [23],
         [25],
         [28],
         [31],
         [40],
         [48],
         [54],
         [67]])
  >>> y
  array([180, 163, 172, 168, 165, 156, 153, 144, 139, 130])
  >>> top_polynomial_regressions(x, y, 20)
  the top 10 polynomial regressions are
  0.9920 Q^2 with degree 7
  0.9838 Q^2 with degree 8
  0.9741 Q^2 with degree 9
  0.9333 Q^2 with degree 18
  0.9332 Q^2 with degree 17
  0.9330 Q^2 with degree 19
  0.9330 Q^2 with degree 15
  0.9327 Q^2 with degree 16
  0.9326 Q^2 with degree 14
  0.9319 Q^2 with degree 13
  >>> top_polynomial_regressions(x, y, 20, 3)
  the top 3 polynomial regressions are
  0.9920 Q^2 with degree 7
  0.9838 Q^2 with degree 8
  0.9741 Q^2 with degree 9
  >>> 
  """
  sorted_polynomial_regressions = lambda x,y, max_degree: sorted([(polynomial_regression(x,y,i)[0], i) for i in range(2, max_degree)], reverse = True)

  print('the top %d polynomial regressions are' %top)
  for regression in sorted_polynomial_regressions(x, y, max_degree)[:top]:
    print('%.4f Q^2 with degree %d' %(regression[0], regression[1]))

def testing_functionality(x, y):
  print('----> Testing:')

  #  '\' is just for to be able to break the line and tabulate nice
  functions = \
  [
    (linear_regression,    linear_regression_output   ),
    (quadratic_regression, quadratic_regression_output),
    (cubic_regression,     cubic_regression_output    )
  ]
  
  print('x = \n', x)
  print('y = '  , y)

  for f_in, f_out in functions:
    r_sq, coefficients, interception = f_in(x,y)
    prediction = \
      lambda age: sum( [coefficients[i] * age**(i + 1) for i in range(len(coefficients))] ) + interception
  
    f_out(r_sq, coefficients, interception, prediction)

  for degree in [7, 8, 9]:
    r_sq, coefficients, interception = polynomial_regression(x, y, degree)
    prediction = \
      lambda age: sum( [coefficients[i] * age**(i + 1) for i in range(len(coefficients))] ) + interception
  
    polynomial_regression_output(r_sq, coefficients, interception, prediction)
  
  top_polynomial_regressions(x, y, 150, 5)

if __name__ == '__main__':
  if len(sys.argv) >= 2 and sys.argv[1] == 'testing':
    testing_functionality(x, y)
  else:
    print('usage: python linear_regression.py testing')


