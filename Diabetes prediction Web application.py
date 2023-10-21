import numpy as np
import pickle
import streamlit as st

# Loading the model
svcModel = pickle.load(open('D:/Learning\Git and Github/Github projects/Diabeses Prediction/trained_model.sav','rb'))

# Making a predicted system
def Diabetes_prediction(input_data):
    # convert input into numpy array
    input_data_array = np.array(input_data)
    # convert 1D input array data into 2D
    input_data_array_2D = input_data_array.reshape(1,-1)
    
    if svcModel.predict(input_data_array_2D)[0]==1:
      return "this person has Diabetes"
    else:
      return "this person doesn't have Diabetes"
      

def main():
    # set web app title
    st.title('Diabetes prediction')
    
    # Get the input data from the user
    Pregnancies=st.text_input('Enter the number of Pregnancies')
    Glucose=st.text_input('Enter the Glucose')
    BloodPressure=st.text_input('Enter the BloodPressure')
    SkinThickness=st.text_input('Enter the SkinThickness')
    Insulin=st.text_input('Enter the Insulin')
    BMI=st.text_input('Enter the BMI')
    DiabetesPedigreeFunction=st.text_input('Enter the DiabetesPedigreeFunction')
    Age=st.text_input('Enter the Age')
    
    # the result
    diagnosis = ''
    
    # check if the button is clicked
    if st.button("Diabetes test Result"):
        diagnosis=Diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    # show the result
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    