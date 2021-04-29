import joblib

# Import the IRIS Dataset to be used in this Kernel
from sklearn.datasets import load_iris  

# Load the Module to split the Dataset into Train & Test 
from sklearn.model_selection import train_test_split

Iris_data = load_iris()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data,Iris_data.target, test_size=0.3,random_state=4)

joblib_file = "joblib_RL_Model.pkl"  

joblib_LR_model = joblib.load(joblib_file)

score = joblib_LR_model.score(Xtest, Ytest)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

print("Predicting a new data from pkl file")
result = joblib_LR_model.predict([[1, 1, 1, 1]])
print(result)
