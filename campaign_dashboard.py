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
    scaler = joblib.load('scaler.pkl')
    cols = joblib.load('model_features.pkl')
    return reg, clf, scaler, cols 

# FIX 2: Correctly unpacking the scaler into the global app state
reg_model, clf_model, scaler, model_cols = load_models()

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
    # 1. Start with a dictionary of all features initialized to 0
    encoded_inputs = {col: 0 for col in model_cols}
    
    # 2. Fill in the base numerical features directly
    encoded_inputs['Duration'] = duration
    encoded_inputs['Impressions'] = impressions
    encoded_inputs['Clicks'] = clicks
    encoded_inputs['Leads'] = leads
    encoded_inputs['Conversions'] = conversions
    encoded_inputs['Acquisition_Cost'] = acq_cost
    encoded_inputs['Engagement_Score'] = eng_score
    
    # 3. Handle your manual channel checkboxes
    encoded_inputs['Email'] = 1 if email else 0
    encoded_inputs['Facebook'] = 1 if facebook else 0
    encoded_inputs['Google'] = 1 if google else 0
    encoded_inputs['Instagram'] = 1 if instagram else 0
    encoded_inputs['WhatsApp'] = 1 if whatsapp else 0
    encoded_inputs['YouTube'] = 1 if youtube else 0

    # 4. Dynamically switch the correct One-Hot Encoded dropdown columns to 1
    if f"Campaign_Type_{campaign_type}" in encoded_inputs:
        encoded_inputs[f"Campaign_Type_{campaign_type}"] = 1
        
    if f"Target_Audience_{target_audience}" in encoded_inputs:
        encoded_inputs[f"Target_Audience_{target_audience}"] = 1
        
    if f"Language_{language}" in encoded_inputs:
        encoded_inputs[f"Language_{language}"] = 1
        
    if f"Customer_Segment_{customer_segment}" in encoded_inputs:
        encoded_inputs[f"Customer_Segment_{customer_segment}"] = 1

    # 5. Convert our cleanly aligned dictionary into a single-row DataFrame
    input_final = pd.DataFrame([encoded_inputs], columns=model_cols)

    # 6. Apply the Scaler matrix safely
    input_final_scaled = scaler.transform(input_final)

    # 7. Make Predictions using completely aligned data
    predicted_revenue = reg_model.predict(input_final_scaled)[0]
    predicted_profit = clf_model.predict(input_final_scaled)[0]
    
    # --- 4. DISPLAY RESULTS ---
    st.divider()
    st.header("🎯 Prediction Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Predicted Revenue", f"₹ {predicted_revenue:,.2f}")

    with res_col2:
        if predicted_profit == 0 or predicted_revenue < 0:
            st.error("Loss-Making Campaign (Loss)")
        else:
            st.success("Profitable Campaign (Profit)")
            
    st.subheader("Key Input Metrics Overview")
    chart_data = pd.DataFrame(
        {"Values": [impressions, clicks, leads, conversions]}, 
        index=["Impressions", "Clicks", "Leads", "Conversions"]
    )
    
    fig = px.bar(chart_data, orientation='h', title="Marketing Funnel Inputs")
    st.plotly_chart(fig, use_container_width=True)
