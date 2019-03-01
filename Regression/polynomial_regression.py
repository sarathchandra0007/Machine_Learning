# Polynomial regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# For correct predections we analyze both regressions

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=4)
X_poly = pol_reg.fit_transform(X, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Graphicla representation
# linear regression

plt.scatter(X,y, color='blue')
plt.plot(X, lin_reg.predict(X), color='red')
plt.title("linear regression")
plt.xlabel("level")
plt.ylabel("salaries")
plt.show()

# ploynomial regression

plt.scatter(X,y, color='blue')
plt.plot(X, lin_reg2.predict(pol_reg.fit_transform(X)), color='red')
plt.title("poly regression")
plt.xlabel("level")
plt.ylabel("salaries")
plt.show()

# custom predections

lin_reg2.predict(pol_reg.fit_transform(12))
