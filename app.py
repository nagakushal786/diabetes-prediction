import streamlit as st
import pickle
import numpy as np

st.title('Diabetes Prediction')
model=pickle.load(open('model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))

col1, col2=st.columns(2)
with col1:
    pregnancies=st.text_input('Number of Pregnancies')
with col2:
    glucose=st.text_input('Glucose Level')

col3, col4 = st.columns(2)
with col3:
    blood_pressure=st.text_input('Blood Pressure value')
with col4:
    skin_thickness=st.text_input('Skin Thickness value')

col5, col6 = st.columns(2)
with col5:
    insulin=st.text_input('Insulin Level')
with col6:
    bmi=st.text_input('BMI value')

col7, col8 = st.columns(2)
with col7:
    diabetes_pedigree_function=st.text_input('Diabetes Pedigree Function value')
with col8:
    age=st.text_input('Age of the Person')

if st.button('Predict'):
    input_data=(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    std_data=scaler.transform(input_data_reshaped)
    prediction=model.predict(std_data)

    if (prediction[0]==0):
        st.write('The Person is not Diabetic')
    else:
        st.write('The Person is Diabetic')