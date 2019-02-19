# Data Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Data
dataset = pd.read_csv('Data.csv')
# Independent variables
x = dataset.iloc[:, :-1].values
# Dependent Variables
y = dataset.iloc[:,  -1].values

# Taking care if missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# Dummy encoder
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Spitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scaling (euqlidian distance)
# feature scaling is not required to Dependendent variables

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

