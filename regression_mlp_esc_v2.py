# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:55:17 2017

@author: Stein
"""

# Import the libraries
print ('[INFO] loading computational backend...')
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)


# Define the base model
def baseline_model():
    # Create the model
    model = Sequential()
    
    model.add(Dense(6, input_dim = 6, 
                    kernel_initializer = 'normal',
                    activation = 'relu'))
    
    model.add(Dense(1, kernel_initializer = 'normal'))
    
    # Compile the model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def large_model():
    # Create the model : 6 inputs -> [13 -> 6] -> 1 output
    model = Sequential()
    
    model.add(Dense(6, input_dim = 6, 
                    kernel_initializer = 'normal',
                    activation = 'relu'))
    
    model.add(Dense(6, 
                    kernel_initializer = 'normal',
                    activation = 'relu'))
    
    model.add(Dense(1, kernel_initializer = 'normal'))
    
    # Compile the model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def wide_model():
    # Create the model : 6 inputs -> [20] -> 1 output
    model = Sequential()
    
    model.add(Dense(20, input_dim = 6, 
                    kernel_initializer = 'normal',
                    activation = 'relu'))
        
    model.add(Dense(1, kernel_initializer = 'normal'))
    
    # Compile the model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def wide_large_model():
    # Create the model : 6 inputs -> [20 -> 6] -> 1 output
    model = Sequential()
    
    model.add(Dense(20, input_dim = 6, 
                    kernel_initializer = 'normal',
                    activation = 'relu'))
    
    model.add(Dense(6, 
                    kernel_initializer = 'normal',
                    activation = 'relu'))
    
    model.add(Dense(1, kernel_initializer = 'normal'))
    
    # Compile the model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model



# Evaluate the model with non-standardized dataset
estimator = KerasRegressor(build_fn = large_model,
                           nb_epoch = 100,
                           batch_size = 5,
                           verbose = 1)

estimator.fit(X_train, y_train, batch_size = 5, epochs = 100)

y_pred = estimator.predict(X_test)

# Evaluate model performance
score = r2_score (y_test_sc, y_pred)



# Evaluate the model with standardize dataset - Base line model
print ('Evaluating baseline model - standardized data...')
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn = baseline_model,
                                        epochs = 50,
                                        batch_size = 5,
                                        verbose = 1)))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits = 10, random_state = 0)
results = cross_val_score(pipeline, X, y, cv = kfold)
kfold_mlp_mean = accuracy_mlp.mean()
kfold_mlp_std = accuracy_mlp.std()