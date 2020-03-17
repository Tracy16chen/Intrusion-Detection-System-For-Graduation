import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score)

# traindata = pd.read_csv('1.csv', header=None)
# testdata = pd.read_csv('2.csv', header=None)
traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)

# X = traindata.iloc[:,0:78]
# Y = traindata.iloc[:,79]
# M = testdata.iloc[:,0:78]
# N = testdata.iloc[:,79]


X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
M = testdata.iloc[:,1:42]
N = testdata.iloc[:,0]


scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(M)
testM = scaler.transform(M)


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testM)
testlabel = np.array(N)



print("LR")
model = LogisticRegression()
model.fit(traindata, trainlabel)

expected = testlabel
predicted = model.predict(testdata)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted , average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("KNN")
model = KNeighborsClassifier()
model.fit(traindata, trainlabel)
print(model)

expected = testlabel
predicted = model.predict(testdata)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted , average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)