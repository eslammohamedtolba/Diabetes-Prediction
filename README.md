# Diabetes Prediction using SVM-SVC  
This Python program uses a Support Vector Machine (SVM) to predict whether a person has diabetes or not based on certain input features. 
It utilizes the scikit-learn library for SVM implementation and includes data preprocessing steps for standardization and train-test splitting.


## Usage
Run the diabetes_prediction.py script to train the SVM model and make predictions: python diabetes_prediction.py

## Dataset
The program uses the "diabetes.csv" dataset, which contains various health-related features and an "Outcome" column indicating whether the person has diabetes (1) or not (0).
The dataset is loaded using pandas and is split into input features (X) and target labels (Y).

## Algorithm
The SVC from SVM algorithm with a linear kernel is employed for binary classification. 
The input features are standardized to ensure consistent scaling. 
The dataset is split into a training set (70%) and a test set (30%) to evaluate the model's accuracy.

## Results
The SVC model achieves an accuracy of 76% on the test set, demonstrating its ability to predict diabetes based on the provided features.

## Making Predictions
You can also make predictions for new data by providing input features. 
Modify the input_data variable in the code to input your own data and run the script to see if the person is predicted to have diabetes or not.

## Contributing
If you would like to contribute to this project, feel free to open issues or submit pull requests.
