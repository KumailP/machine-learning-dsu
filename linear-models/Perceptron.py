# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 01:40:05 2019

@author: Kumail
"""

import pandas
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Perceptron

dataset = pandas.read_csv("../datasets/iris.csv")
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.20
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

clf = Perceptron(tol=1e-3, random_state=seed).fit(X_train, Y_train)

predict = clf.predict(X_test[:5, :])
actual = Y_test[:5]
scoreTest = round(clf.score(X_test, Y_test)*100, 1)
scoreTrain= round(clf.score(X_train, Y_train)*100, 1)

print(f"Prediction: {predict}")
print(f"Actual: {actual}")
print(f"Training Score: {scoreTrain}%")
print(f"Test Score: {scoreTest}%")