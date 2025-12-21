import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="SilviaMartin/Visitwithus", filename="visit_with_us_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism package  Prediction App")
st.write("The App predicts whether customers will purchase the newly introduced Wellness Tourism Package before contacting them")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Customer Age", min_value=18, max_value=61, value=40)
CityTier = st.number_input("The city tier of customer ", min_value=1, max_value=3, value=2)
DurationOfPitch = st.number_input("The duration of the pitch",min_value=5,max_value=127,value=20)
NumberOfPersonVisiting = st.number_input("Number of persons visiting", min_value=1, max_value=5, value=2)
NumberOfFollowups = st.number_input("Number of followups", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.number_input("The preferred property star rating", min_value=3, max_value=5, value=3)
NumberOfTrips = st.number_input("Number of trips", min_value=1, max_value=22, value=2)
Passport = st.radio("Having Passport?", [0, 1])
OwnCar=st.radio("Having Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of children visiting", min_value=0, max_value=3, value=2)
MonthlyIncome=st.number_input("Specify the monthly income",min_value=1000,max_value=100000,value=3000)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
TypeofContact=st.selectbox("Mention the type of contact", ["Company Invited","Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Free Lancer","Large Business","Salaried","Small Business"])
Gender  = st.selectbox("Customer Gender", ["Female","Male"])
ProductPitched=st.selectbox("Mention the product pitched to customer", ["Basic","Deluxe","Standard","King","Super Deluxe"])
MaritialStatus=st.selectbox("Mention the marital status of customer", ["Single","Married","Unmarried","Divorced"])
Designation=st.selectbox("Mention the designation of customer", ["Executive","Manager","Senior Manager","VP","AVP"])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips':NumberOfTrips,
    'Passport':Passport,
    'OwnCar':OwnCar,
    'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
    'MonthlyIncome':MonthlyIncome,
    'PitchSatisfactionScore':PitchSatisfactionScore,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritialStatus,
    'Designation': Designation
  }])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Product Purchased" if prediction == 1 else "Not Purchased"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
