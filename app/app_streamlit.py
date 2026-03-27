import streamlit as st
import pandas as pd
import joblib

st.title("ASG 05 MD - Michael - Spaceship Titanic Model Deployment")

model = joblib.load("models/model.pkl")

st.header("Passenger Information")

PassengerId = st.text_input("PassengerId","0001_01")
HomePlanet = st.selectbox("HomePlanet",["Earth","Europa","Mars"])
CryoSleep = st.selectbox("CryoSleep",[True,False])
Cabin = st.text_input("Cabin","B/0/P")
Destination = st.selectbox("Destination",["TRAPPIST-1e","PSO J318.5-22","55 Cancri e"])
Age = st.number_input("Age",0,100,30)
VIP = st.selectbox("VIP",[True,False])

RoomService = st.number_input("RoomService",0)
FoodCourt = st.number_input("FoodCourt",0)
ShoppingMall = st.number_input("ShoppingMall",0)
Spa = st.number_input("Spa",0)
VRDeck = st.number_input("VRDeck",0)
Name = st.text_input("Name","Michael Yeremia")

input_df = pd.DataFrame({
    "PassengerId":[PassengerId],
    "HomePlanet":[HomePlanet],
    "CryoSleep":[CryoSleep],
    "Cabin":[Cabin],
    "Destination":[Destination],
    "Age":[Age],
    "VIP":[VIP],
    "RoomService":[RoomService],
    "FoodCourt":[FoodCourt],
    "ShoppingMall":[ShoppingMall],
    "Spa":[Spa],
    "VRDeck":[VRDeck]
})

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    if prediction:
        st.success("Passenger Transported ✅")
    else:
        st.error("Passenger Not Transported ❌")


