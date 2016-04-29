# This file is to make submission to the digit recognizer project on kaggle.com. It contains three simple algorithms: LinearRegression, LogisticRegression,
# and RandomForestClassifier. They are all simply implemented using the sklearn package in anacoda.  

import csv, pandas, math, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
start = time.time()

def quantize(func):
	for i, ele in enumerate(func):
		if ele < 0:
			func[i] = 0
		elif ele >= 10:
			func[i] = 9
		else:
			func[i] = math.floor(ele)
	return func.astype(int)
	
def Visual1(train2,predictions):
	values = np.zeros( (10,10) )

	for i in range( 10 ):
		for j in range( 10 ):
			filt = (train2['label'] == i) & (predictions == j)
			values[i,j] = len(filt[ filt==True ])

	return values	

train_file = pandas.read_csv('train.csv')
f = open('train.csv','r')
dat = f.readlines()
col_label = dat[0].split(',')
image = np.zeros( (len(dat[1:]), len(col_label)) )
for i in range( 1,len(dat[1:]) ):
	image[i,:] = dat[i].split(',')
	
image = image.astype(int)



train1 = train_file.loc[:30000]
#label = list(train1['label'])
#train1[ train1<15 ] = 0
#train1 = image[:30000,:]
#train2 = image[30001:,:]
train2 = train_file.loc[30001:]


#test_file = pandas.read_csv("test.csv")

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
#alg.fit(train_file[predictors], train_file["label"])
alg.fit(train1[predictors],train1["label"])
#alg.fit(train1[:,1:],train1[:,0])

print("starting predicting")
#predictions = alg.predict(test_file[predictors])
predictions = alg.predict(train2[predictors])
#predictions = alg.predict(train2[:,1:])

error = predictions - train2['label']



# if one uses linearregression one gets floating number predictions, and then one has to use the loop below to quantize the predictions
#predictions = quantize(predictions)
	
print(predictions[0:5],type(predictions[1]),len(predictions))
# Create a new dataframe with only the columns Kaggle wants from the dataset.
#submission = pandas.DataFrame({
#        'ImageId': range(1,28001),
#        'Label': predictions
#    })
#submission.to_csv("submission.csv", index=False)	
#print(submission.head(5))


			
values = Visual1(train2,predictions)
values[ values == 0 ] = 1
plt.imshow(values,origin='lower',interpolation='none',norm=LogNorm(),cmap=plt.cm.hot)
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.colorbar()
plt.show()

frac_correct = len( error[error==0] )/len(train2)
stop = time.time()
print(frac_correct, "Time %s"%(stop-start))