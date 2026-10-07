import streamlit as st
import pandas as pd
import joblib

# Load trained classifier and scaler
classifier = joblib.load("cluster_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Dictionary mapping cluster numbers to descriptive business statements
CLUSTER_LABELS = {
    0: "Low Income / Low Spender (Budget Customer)",
    1: "High Income / Big Spender (VIP Customer)",
    2: "Average Income / Active Web Shopper",
    3: "Low Income / Occasional Spender",
    4: "High Income / High Wealth Outlier"
}

st.title("Customer Segmentation Real-Time Prediction")
st.write("Enter customer details to predict their cluster segment.")

# Input fields matching training features
income = st.number_input("Income ($)", min_value=0, max_value=200000, value=50000)
recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=30)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_catalog_purchases = st.number_input("Number of Catalog Purchases", min_value=0, max_value=100, value=5)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
total_spending = st.number_input("Total Spending ($)", min_value=0, max_value=10000, value=1000)

if st.button("Predict Segment"):
    # Construct input dataframe
    input_df = pd.DataFrame([{
        "Income": income,
        "Recency": recency,
        "NumWebPurchases": num_web_purchases,
        "NumCatalogPurchases": num_catalog_purchases,
        "NumWebVisitsMonth": num_web_visits,
        "Total_spending": total_spending
    }])

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict numeric cluster ID
    predicted_cluster = classifier.predict(input_scaled)[0]
    
    # Map cluster ID to descriptive label
    segment_statement = CLUSTER_LABELS.get(predicted_cluster, "Unknown Segment")

    # Display result
    st.success(f"**Predicted Segment:** {segment_statement}")
    st.info(f"**Cluster ID:** Cluster {predicted_cluster}")
