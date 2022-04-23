# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:55:58 2022

@author: Amit
"""


import streamlit as st
import joblib


def main():
    html_temp = """
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black";text-align:center> House Price Predictor App by Amit Kumar</h2>
    </div>
    
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    model = joblib.load('model_joblib_gr2')
    
    p1 = st.number_input("Enter per capita crime")
    p2 =st.number_input("proportion of residential land zoned for lots over 25,000 sq.ft.")
    p3 = st.number_input("proportion of non-retail business acres per town")
    s1=st.selectbox("Charles River",("Beside Charles river","Away from Charles river"))
    if s1=="Beside Charles river":
        p4=1
    else:
        p4=0
        
    p5 = st.number_input("nitric oxides concentration")
    p6 = st.number_input("average number of rooms")
    p7 = st.number_input("proportion of owner-occupied units built prior to 1940 [age]")
    p8 = st.number_input("weighted distances to five Boston employment centres")
    p9 = st.number_input("index of accessibility to radial highways")
    p10 = st.number_input("full-value property-tax rate per $10,000 [tax]")
    p11= st.number_input("Epupil-teacher ratio")
    p12 = st.number_input("proportion of blacks by town [blacks]")
    p13= st.number_input("percentage of lower status of the population [lstat]")
    
    if st.button('Predict'):
        prediction = model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
        st.balloons()
        st.success('Price of house in $1000s is {} '.format(round(prediction[0],2)))
    

    
    
if __name__ == '__main__':
        main()
        
       