#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix,classification_report
from sklearn.externals import joblib



_data=np.load('NBx.npy')
_label=np.load('NBy.npy')

N = np.shape(_data)[0]
train_test_split_percentage = 0.75

X_train = _data[:int(N * train_test_split_percentage),:]
print(X_train.shape)
X_test = _data[int(N * train_test_split_percentage):,:]
print(X_test.shape)

Y_train = _label[:int(N * train_test_split_percentage)]
print(Y_train.shape)
Y_test = _label[int(N * train_test_split_percentage):]
print(Y_test.shape)

pipeline = make_pipeline(RandomForestClassifier())

# Add a dict of estimator and estimator related parameters in this list
hyperparameters = {
                'randomforestclassifier__n_estimators': [25,50,75,100],
                'randomforestclassifier__max_features' : [None, "log2", "auto"]
                }



# # 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=5,verbose=1,n_jobs=-1)
clf.fit(X_train, Y_train)


print(clf.best_params_)
print(clf.best_estimator_)
# print(clf.cv_results_ )

print(clf.best_score_ )
print (clf.refit)
 
# 9. Evaluate model pipeline on test data
pred = clf.predict(X_test)


print(accuracy_score(Y_test, pred))

cm = confusion_matrix(Y_test, pred)
print(classification_report(Y_test, pred))
print(cm)