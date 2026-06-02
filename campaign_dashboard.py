import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# --- 1. LOAD MODELS & FEATURES ---
@st.cache_resource
def load_models():
    reg = joblib.load('revenue_regression_model.pkl')
    clf = joblib.load('profit_classification_model.pkl')
    cols = joblib.load('model_features.pkl')
    return reg, clf, cols

reg_model, clf_model, model_cols = load_models()

# --- 2. BUILD THE UI ---
st.title("📈 Marketing Campaign Performance Predictor")
st.markdown("Enter the campaign details below to predict expected Revenue and whether the campaign will be Profitable.")

st.sidebar.header("Campaign Settings")

# Numerical Inputs
duration = st.sidebar.slider("Duration (Days)", 1, 365, 30)
impressions = st.sidebar.number_input("Expected Impressions", min_value=0, value=10000)
clicks = st.sidebar.number_input("Expected Clicks", min_value=0, value=500)
leads = st.sidebar.number_input("Expected Leads", min_value=0, value=50)
conversions = st.sidebar.number_input("Expected Conversions", min_value=0, value=10)
acq_cost = st.sidebar.number_input("Acquisition Cost (Spend)", min_value=0, value=1500)
eng_score = st.sidebar.slider("Target Engagement Score", 0.0, 100.0, 50.0)

# Categorical Inputs
campaign_type = st.selectbox("Campaign Type", ["Social Media", "Paid Ads", "Influencer", "Email"])
target_audience = st.selectbox("Target Audience", ["College Students", "Tier 2 City Customers", "Youth", "Working Women", "HNI"])
language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Telugu", "Marathi"])
customer_segment = st.selectbox("Customer Segment", ["New", "Returning", "Premium"])

# Multi-Label Channel Inputs
st.markdown("### Marketing Channels")
col1, col2, col3 = st.columns(3)
with col1:
    email = st.checkbox("Email")
    facebook = st.checkbox("Facebook")
with col2:
    google = st.checkbox("Google")
    instagram = st.checkbox("Instagram")
with col3:
    whatsapp = st.checkbox("WhatsApp")
    youtube = st.checkbox("YouTube")

# --- 3. PREDICTION LOGIC ---
if st.button("Predict Campaign Performance"):
    # Create a dataframe for the input
    input_data = pd.DataFrame({
        'Duration': [duration],
        'Impressions': [impressions],
        'Clicks': [clicks],
        'Leads': [leads],
        'Conversions': [conversions],
        'Acquisition_Cost': [acq_cost],
        'Engagement_Score': [eng_score],
        'Campaign_Type': [campaign_type],
        'Target_Audience': [target_audience],
        'Language': [language],
        'Customer_Segment': [customer_segment],
        'Email': [1 if email else 0],
        'Facebook': [1 if facebook else 0],
        'Google': [1 if google else 0],
        'Instagram': [1 if instagram else 0],
        'WhatsApp': [1 if whatsapp else 0],
        'YouTube': [1 if youtube else 0]
    })

    # One-Hot Encode categorical features just like training
    input_encoded = pd.get_dummies(input_data, columns=['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment'])

    # Align columns with the model features (fill missing with 0)
    # This prevents errors if a specific category wasn't selected in the dropdown
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Reorder columns to match training exactly
    input_final = input_encoded[model_cols]

   # Make Predictions
    predicted_revenue = reg_model.predict(input_final)[0]
    predicted_profit = clf_model.predict(input_final)[0]

    # --- 4. DISPLAY RESULTS ---
    st.divider()
    st.header("🎯 Prediction Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Predicted Revenue", f"₹ {predicted_revenue:,.2f}")

    with res_col2:
        # Perfectly indented and using the correct 'predicted_profit' variable
        if predicted_profit == 0 or predicted_revenue < 0:
            st.error("Loss-Making Campaign (Loss)")
        else:
            st.success("Profitable Campaign (Profit)")
    # Display Key Input Visualization (Make sure this is indented!)
    st.subheader("Key Input Metrics Overview")
    chart_data = pd.DataFrame(
        {"Values": [impressions, clicks, leads, conversions]}, 
        index=["Impressions", "Clicks", "Leads", "Conversions"]
    )
    
    # Create a Plotly bar chart (Indented 4 spaces inside the button block)
    fig = px.bar(chart_data, orientation='h', title="Marketing Funnel Inputs")
    st.plotly_chart(fig, use_container_width=True)