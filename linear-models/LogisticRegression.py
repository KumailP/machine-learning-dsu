# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:41:11 2019

@author: Kumail
"""

import pandas
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

dataset = pandas.read_csv("../datasets/iris.csv")
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.29
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs', multi_class='multinomial').fit(X_train, Y_train)

predict = clf.predict(X_test[:5, :])
actual = Y_test[:5]
scoreTest = round(clf.score(X_test, Y_test)*100, 1)
scoreTrain= round(clf.score(X_train, Y_train)*100, 1)

print(f"Prediction: {predict}")
print(f"Actual: {actual}")
print(f"Training Score: {scoreTrain}%")
print(f"Test Score: {scoreTest}%")

sepal_length = int(input("Enter sepal length: ")) 
sepal_width = int(input("Enter sepal width: "))
petal_length = int(input("Enter petal length: "))
petal_width = int(input("Enter petal width: "))
inputData = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1,-1)

newPredict = clf.predict(inputData)[0]
print(f"Prediction is: {newPredict}")