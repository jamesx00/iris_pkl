# Import Required packages 
#-------------------------

# Import the Logistic Regression Module from Scikit Learn
from sklearn.linear_model import LogisticRegression  

# Import the IRIS Dataset to be used in this Kernel
from sklearn.datasets import load_iris  

# Load the Module to split the Dataset into Train & Test 
from sklearn.model_selection import train_test_split

Iris_data = load_iris()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data,Iris_data.target, test_size=0.3,random_state=4)

LR_Model = LogisticRegression(C=0.1, max_iter=20,fit_intercept=True, n_jobs=3, solver='liblinear')
LR_Model.fit(Xtrain, Ytrain)

import joblib

joblib_file = "joblib_RL_Model.pkl"  

joblib.dump(LR_Model, joblib_file)

score = joblib_LR_model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  
