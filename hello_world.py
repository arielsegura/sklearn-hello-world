# Based on https://scikit-learn.org/stable/tutorial/basic/tutorial.html

from sklearn import datasets
from sklearn import svm

print("Running sklearn Hello World!")

# set up
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)
# A dataset is a dictionary-like object that holds all the data and some metadata about the data.
# This data is stored in the .data member, which is a n_samples, n_features array.
# In the case of supervised problem, one or more response variables are stored in the .target member.

# The data is always a 2D array, shape (n_samples, n_features), although the original data may have had a different shape.

print("Building classifier")
clf = svm.SVC(gamma=0.001, C=100.) # TODO add grid search and cross validation.

# some data cleaning if needed

# training
print("Fitting classifier")
clf.fit(digits.data[:-1], digits.target[:-1])

# predict
print("Predicting")
prediction = clf.predict(digits.data[-1:])

print(prediction)
print("Real Value ", digits.target[-1:])
