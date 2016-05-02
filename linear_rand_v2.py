# This file is to make submission to the digit recognizer project on kaggle.com. It contains three simple algorithms: LinearRegression, LogisticRegression,
# and RandomForestClassifier. They are all simply implemented using the sklearn package in anacoda.  

import csv
import pandas
import numpy
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def quantize(func):
	for i, ele in enumerate(func):
		if ele < 0:
			func[i] = 0
		elif ele >= 10:
			func[i] = 9
		else:
			func[i] = math.floor(ele)
	return func.astype(int)
	
train_file = pandas.read_csv('train.csv')
test_file = pandas.read_csv("test.csv")

# The columns we'll use to predict the target
predictors = []
for ele in range(0,783):
	predictors.append('pixel'+str(ele))

#cross validation phase
#print('starting cross validation')


# Initialize our algorithm

#alg = LogisticRegression(random_state=1)          # LogisticRegression doesn't work somehow
#alg = LinearRegression()         #LinearRegression has an accuracy around 0.26671
alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=3, min_samples_leaf=1) # RandomForestClassifier has around 0.95


print("starting learning")
alg.fit(train_file[predictors], train_file["label"])

print("starting predicting")
predictions = alg.predict(test_file[predictors])

# if one uses linearregression one gets floating number predictions, and then one has to use the loop below to quantize the predictions
#predictions = quantize(predictions)
	
print(predictions[0:5],type(predictions[1]),len(predictions))
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        'ImageId': range(1,28001),
        'Label': predictions
    })
submission.to_csv("submission.csv", index=False)	
#print(submission.head(5))

