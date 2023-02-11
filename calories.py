# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:09:25 2022

@author: mihir
"""

import xgboost as xg
import streamlit as st
import numpy as np

#loading the model
m = xg.XGBRegressor()
m.load_model("C:/Users/mihir/Calories Prediction/calories_pred.model")


st.title('Calories Burnt Predictor')
gender_names = ['Male', 'Female']
Gender = st.radio('Gender',gender_names)

col1,col2 = st.columns(2)
with col1:
    Age = st.text_input('Age')
    Weight = st.text_input('Weight')
    Heart_Rate = st.text_input('Heart rate')
    
with col2:
    Height = st.text_input('Height')
    Duration = st.text_input('Duration of exercise')
    Body_Temp = st.text_input('Body Temperature')


#prediction
result = ''


if st.button('Predict Calories Burnt'):
   if Gender == 'Male':
       lis = (0, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
       input_data = np.array(lis, dtype=object)
       pred = m.predict([input_data])
       result = pred[0]
   else:
       lis = (1, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
       input_data = np.array(lis, dtype=object)
       pred = m.predict([input_data])
       result = pred[0]
   
st.success(result)
   
