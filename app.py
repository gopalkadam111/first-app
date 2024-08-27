import streamlit as st
import pandas as pd
import pickle
import datetime

st.header("Car24 Used Car Price Prediction")
df = pd.read_csv("./cars_data.csv")

st.dataframe(df.head(2), use_container_width=True)

with open('./car_pred','rb') as f:
    model = pickle.load(f)

col1, col2 = st.columns(2)

with col1:
    fuel_type = col1.selectbox("Select Fuel Type",df.fuel_type.unique())
    transmission_type = col1.selectbox("Select Transmission Type",df.transmission_type.unique())
with col2: 
    engine = col2.slider("Select engine power",min_value=int(df.engine.min()),max_value= int(df.engine.max()),step=100)
    seats = col2.slider("Select no. of seats",min_value=df.seats.min(),max_value= df.seats.max(),step=1)
fuel_types = {
    "Petrol": 1,
    "Diesel": 2,
    "CNG": 3,
    "LPG": 4,
    "Electric": 5
}

transmission_types = {
    "Manual": 1,
    "Automatic": 2
}



fuel_type = fuel_types[fuel_type]
transmission_type = transmission_types[transmission_type]
input_features = [[2018,1,65000,fuel_type, \
                   transmission_type,18.5,engine,50,seats]]

if st.button("Predict"):
    price = model.predict(input_features)
    st.write(f"The price of car is {round(price[0],2)}")









