# May 12, 2016
# A rough version of an n-layer nerual network algorithm, using gradient descent
# The Algorithm is written as a class. 
# An example for the titanic competition on kaggle.com is presented. The accuracy with the test parameters is around 70%
# 
import csv
import pandas
import numpy as np

print('finished importing')
class nnet:
    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)
        
    def multiplecost(self, X, y):
        # feed through network
        l1, l2 = self.feedforward(X) 
        # compute error
        inner = y * np.log(l2) + (1-y) * np.log(1-l2)
        # negative of average error
        return -np.mean(inner)
    
    def feedforward(self, X):
        # feedforward to the first layer
        l1 = sigmoid_activation(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2
    
    def predict(self, X):
        _, y = self.feedforward(X)
        return y
	
    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))
        
        self.costs = []
        cost = self.multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres+1  # set an inital costprev to past while loop
        counter = 0  # intialize a counter

        # Loop through until convergence
        for counter in range(self.maxepochs):
            # feedforward through network
            l1, l2 = self.feedforward(X)

            # Start Backpropagation
            # Compute gradients
            l2_delta = (y-l2) * l2 * (1-l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate
            
            # Store costs and check for convergence
            counter += 1  # Count
            costprev = cost  # Store prev cost
            cost = self.multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
                break

def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))
	
# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 1000000       
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.000001  
# Number of hidden units
hidden_units = 8

# Initialize model 
model = nnet(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model
#model.learn(X, y)

# Now we would start a test for this algorithm on the titanic competition. One should have the data in order to run it. 
# Start loading the data

titanic = pandas.read_csv('train.csv')
titanic["ones"] = np.ones(titanic.shape[0])
titanic["Age"] = titanic["Age"].fillna(int(titanic["Age"].median()))
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

titanic_test = pandas.read_csv("test.csv")
titanic_test["ones"] = np.ones(titanic_test.shape[0])
titanic_test["Age"] = titanic_test["Age"].fillna(int(titanic["Age"].median()))
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

print('begin loading files')

# Choose your features
predictors = ['ones','Age','Sex','Embarked']

X_train = titanic[predictors].values.astype(int)
X_test = titanic_test[predictors].values.astype(int)
y_train = titanic['Survived'].values.astype(int) 

# Splitting the training data into train (1:600) and test (601:891) part
X_test = X_train[600:]
X_train = X_train[:601]
y_test = y_train[600:]
y_train = y_train[:601]

print('learning')
#model.learn(train_file[predictors], train_file["label"])
model.learn(X_train, y_train)

print('predicting')
predictions = model.predict(X_test)

# Converting predictions to 0/1
for i, ele in enumerate(predictions[0]):
	if ele <= np.mean(predictions[0]):
		predictions[0][i] = 0
	else:
		predictions[0][i] = 1
		

print(predictions[0])

# Getting accuracy
acu = 0
for i, ele in enumerate(predictions[0]):
	if ele == y_test[i]:
		acu += 1

acu = acu/len(predictions[0])
print(acu)

# Uncomment things below if you want to make a submission
"""
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions[0]
    })

submission.to_csv("submission_NNet.csv", index=False)	
"""