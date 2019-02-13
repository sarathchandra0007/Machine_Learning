# Data Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Data
dataset = pd.read_csv('Data.csv')
# Dependent variables
x = dataset.iloc[:, :-1].values
#Independent Variables
y = dataset.iloc[:,  -1].values

# Taking care if missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
