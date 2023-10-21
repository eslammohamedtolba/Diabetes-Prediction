# import required modules for the prediction process
import numpy as np, pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# uploading Diabetes Dataset
Diabetes_data = pd.read_csv("diabetes.csv")


# showing the Dataset
Diabetes_data.head()
# number of rows and columns
Diabetes_data.shape
# find all statistical info about the diabetes dataset
Diabetes_data.describe()
# count all states that have a Diabetes and not have
Diabetes_data['Outcome'].value_counts()
# find relation between output and all inputs
Diabetes_data.groupby('Outcome').mean()


# split Diabetes data as input and output
X = Diabetes_data.drop(columns='Outcome',axis=1)
Y = Diabetes_data['Outcome']


# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)


# apply algorithm on input and output to predict
svcModel = svm.SVC(kernel='linear')
svcModel.fit(x_train,y_train)


# find the difference between test output and predicted output
for value in zip(svcModel.predict(x_test),y_test):
    print(value)
accuracy_score_train = accuracy_score(svcModel.predict(x_train),y_train)
accuracy_score_test = accuracy_score(svcModel.predict(x_test),y_test)
print(accuracy_score_train,accuracy_score_test)



# Making a predicted system

input_data = (1,189,60,23,846,30.1,0.398,59) # input user data
# convert input into numpy array
input_data_array = np.array(input_data)
# convert 1D input array data into 2D
input_data_array_2D = input_data_array.reshape(1,-1)

if svcModel.predict(input_data_array_2D)[0]==1:
    print("this person has Diabetes")
else:
    print("this person doesn't have Diabetes")



# Importing pickle module to help in saving the model
import pickle as pk
# Saving the model
file_name = 'trained_model.sav'
pk.dump(svcModel , open(file_name,'wb'))
