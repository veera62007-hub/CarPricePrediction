import streamlit as st
import joblib
import pandas as pd

model = joblib.load('car_price_model.pkl')

st.title('Car Price Prediction')

km = st.number_input('Kilometers Driven', min_value=0, value=50000)
year = st.number_input('Manufacturing Year', min_value=1990, max_value=2026, value=2015)
fuel = st.selectbox('Fuel Type', ['CNG', 'Diesel', 'LPG', 'Petrol'])

if st.button('Predict Price'):
    age = 2026 - year
    data = pd.DataFrame([[km, age, fuel == 'Diesel', fuel == 'LPG', fuel == 'Petrol']],
                        columns=['km_driven', 'age', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol'])
    price = model.predict(data.astype(float))
    st.success(f'Predicted Price: {price[0]:.2f}')
