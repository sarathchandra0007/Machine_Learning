import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fit Simple linear regression model to training set

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Predecting test results
y_pred = linear_regression.predict(X_test)

# Visualizing training set results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linear_regression.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')

# Visualizing test set results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, linear_regression.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')

