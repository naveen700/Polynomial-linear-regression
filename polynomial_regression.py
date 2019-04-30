# Polynomial Regression
# Business Problem :-
# Hiring new employee(new employee said his salary was 160k at last company and expecting more than that in new company),
# HR want to make sure whether he is right or bluffing about his salary by using our regression model. 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
 

# Fitting Polynomial Regression to the dataset
# we need libaray which can add polynomial terms to the linear equation ,class name is PolynomialFeatures from preprocessing package
from sklearn.preprocessing import PolynomialFeatures
#poly_reg will convert our X(consist of simple variables) into X_poly(consists of powers of variablel like x , x^2, x^3,x^4) like a polynomial equation..
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


# visualizing the linear regression model
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Truth or bluff  (Linear Regression) ')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# visualizing the polynomial Regressor results
X_grid = np.arange(min(X),max(X),0.1)
# arange will return vector ,we need to convert the vector into matrix
X_grid = X_grid.reshape(len(X_grid) ,1)

plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluff  (Polynomial Regression) ') 
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# checking person is bluffing or not.
# predicting the new result with Linear Regression.
data = [6.5]
data = np.array(data)
data = data.reshape(1,-1)
lin_reg.predict(data)
# predicting the new result with  polynomial regression.
lin_reg2.predict(poly_reg.fit_transform(data))







