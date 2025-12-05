# Import necessary libraries for web app, data handling, and model loading
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained fraud detection model
model = joblib.load('fraud_detection_model.pkl')

# Set up the web app title and instructions
st.title("Fraud Detection Model")
st.markdown(" Please input the transaction details to predict if it's fraudulent or not. ") 

st.divider()

# Input section: Collect transaction details from the user
transaction_type = st.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)" , min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Prediction logic: triggered when user clicks the button
if st.button("Predict Fraud"):
    # Calculate derived features that the model expects
    balancedDifforiginal = newbalanceOrig - oldbalanceOrg
    balanceDiffDestination = newbalanceDest - oldbalanceDest
    
    # Prepare input data in DataFrame format (matching model's expected format)
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balancedDifforiginal": balancedDifforiginal,
        "balanceDiffDestination": balanceDiffDestination
    }])
    
    # Get prediction from the model (0 = Not Fraudulent, 1 = Fraudulent)
    prediction = model.predict(input_data)[0]

    # Display the prediction result
    st.subheader(f"Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'}")

    # Show a warning if the transaction is predicted to be fraudulent
    if prediction == 1:
        st.error("Warning: This transaction is predicted to be fraudulent!")
    else:
        st.success("This transaction is predicted to be legitimate.")