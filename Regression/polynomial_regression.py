# Polynomial regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


from sklearn.cross_validation import train_test_split
