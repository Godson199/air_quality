import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import pickle  

# loading model
pickle_in = open("air_quality_classifier.pkl", "rb")
model = pickle.load(pickle_in)

# loading model
pickle_in = open("air_quality_classifier2.pkl", "rb")
model_2 = pickle.load(pickle_in)

# loading model
pickle_in = open("air_quality_classifier3.pkl", "rb")
model_3 = pickle.load(pickle_in)

# side_bar to navigate
with st.sidebar:
    selected = option_menu("Multiple Model Prediction",
    ["XGBoost Classifier",
    "Logistic Regression Classifier",
    "LGBM Classifier"],
    icons=["1-circle-full", "2-circle-full", "3-circle-full"],
    default_index=0)



#features = ['Temperature', 'Relative_Humidity', 'Sensor1_PM2_5', 'Sensor2_PM2_5',\
#              'Datetime_month', 'Datetime_year', 'Datetime_hour',  'Datetime_day']

# making prediction with the model
def predict_air_quality(Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,Datetime_year, Datetime_hour,Datetime_day):

    prediction = model.predict(np.array([[Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,	Datetime_year, Datetime_hour,Datetime_day]], dtype=object))
    return prediction

def predict_air_quality_2(Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,Datetime_year, Datetime_hour,Datetime_day):

    prediction = model_2.predict(np.array([[Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,	Datetime_year, Datetime_hour,Datetime_day]], dtype=object))
    return prediction

def predict_air_quality_3(Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,Datetime_year, Datetime_hour,Datetime_day):

    prediction = model_3.predict(np.array([[Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,	Datetime_year, Datetime_hour,Datetime_day]], dtype=object))
    return prediction

# building interface with streamlit
def main():
    #st.title('Air Quality Predictor')
    html_temp = """
    <div style = "background-color:tomato;padding:100px>
    <h2 style="color:white;text-align:centre;">Streamlit Air Quality ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    if (selected == "XGBoost Classifier"):
        st.title("Air Quality Prediction Using XGBoost Classifier")

        col1, col2, col3 = st.columns(3)

        with col1:
            Temperature = st.text_input("Temperature", "")
        with col2:
            Relative_Humidity = st.text_input("Relative_Humidity", "")

        with col3:
            Sensor1_PM2_5 = st.text_input("Sensor1_PM2.5", "")
        
        with col1:
            Sensor2_PM2_5 = st.text_input("Sensor2_PM2.5", "")
        
        with col2:
            Datetime_month = st.text_input("Datetime_month", "")
        
        with col3:
            Datetime_year = st.text_input("Datetime_year", "")
        
        with col1:
            Datetime_hour = st.text_input("Datetime_hour", "")

        with col2:
            Datetime_day = st.text_input("Datetime_day", "")

        result=""
        if st.button("Air Quality"):
            result = predict_air_quality(Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,Datetime_year, Datetime_hour,Datetime_day)

            if result == 1:
                st.success("Good air Quality ")
            else:
                st.success("Faulty Sensor, air will be Flagged!")
    
    # Next Model

    if (selected == "Logistic Regression Classifier"):
        st.title("Air Quality Prediction Using Logistic Regressor")

        col1, col2, col3 = st.columns(3)

        with col1:
            Temperature = st.text_input("Temperature", "")
        with col2:
            Relative_Humidity = st.text_input("Relative_Humidity", "")

        with col3:
            Sensor1_PM2_5 = st.text_input("Sensor1_PM2.5", "")
        
        with col1:
            Sensor2_PM2_5 = st.text_input("Sensor2_PM2.5", "")
        
        with col2:
            Datetime_month = st.text_input("Datetime_month", "")
        
        with col3:
            Datetime_year = st.text_input("Datetime_year", "")
        
        with col1:
            Datetime_hour = st.text_input("Datetime_hour", "")

        with col2:
            Datetime_day = st.text_input("Datetime_day", "")

        result=""
        if st.button("Air Quality"):
            result = predict_air_quality_2(Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,Datetime_year, Datetime_hour,Datetime_day)

            if result == 1:
                st.success("Good air Quality ")
            else:
                st.success("Faulty Sensor, air will be Flagged!")

    if (selected == "LGBM Classifier"):
        st.title("Air Quality Prediction LGBM Classifier")
        col1, col2, col3 = st.columns(3)

        with col1:
            Temperature = st.text_input("Temperature", "")
        with col2:
            Relative_Humidity = st.text_input("Relative_Humidity", "")

        with col3:
            Sensor1_PM2_5 = st.text_input("Sensor1_PM2.5", "")
        
        with col1:
            Sensor2_PM2_5 = st.text_input("Sensor2_PM2.5", "")
        
        with col2:
            Datetime_month = st.text_input("Datetime_month", "")
        
        with col3:
            Datetime_year = st.text_input("Datetime_year", "")
        
        with col1:
            Datetime_hour = st.text_input("Datetime_hour", "")

        with col2:
            Datetime_day = st.text_input("Datetime_day", "")

        result=""

        if st.button("Air Quality"):
            result = predict_air_quality_3(Temperature,Relative_Humidity,Sensor1_PM2_5,Sensor2_PM2_5,Datetime_month,Datetime_year, Datetime_hour,Datetime_day)

            if result == 1:
                st.success("Good air Quality ")
            else:
                st.success("Faulty Sensor, air will be Flagged!")

    if st.button("About"):
        st.text("Built by: Izuogu Chibuzor Godson")


if __name__== "__main__":
    main()
