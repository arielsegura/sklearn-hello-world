# Based on https://scikit-learn.org/stable/tutorial/basic/tutorial.html

import matplotlib
from sklearn import datasets
from sklearn import svm

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

print("Running sklearn Hello World!")

# set up
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
digit_to_test = 1
some_digit = digits.data[digit_to_test]
digit_target = digits.target[digit_to_test]

prediction = clf.predict([some_digit])

print(prediction)
print("Real Value ", digit_target)

some_digit_image = some_digit.reshape(8, 8)
print("Rendering")

plt.imshow(some_digit_image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.axis("off")
plt.show()