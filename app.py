import streamlit as st
import pandas as pd 
import numpy as np 
import joblib 

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation Analysis App")
st.write("Enter customer data to predict the segment")

# Add sample data buttons
st.write("### 🚀 Quick Test with Sample Data:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💰 High Spender"):
        st.session_state.age = 50
        st.session_state.income = 80000
        st.session_state.total_spending = 2000
        st.session_state.num_web_purchases = 15
        st.session_state.num_store_purchases = 10
        st.session_state.num_web_visits = 2
        st.session_state.recency = 15

with col2:
    if st.button("💡 Budget Conscious"):
        st.session_state.age = 25
        st.session_state.income = 30000
        st.session_state.total_spending = 200
        st.session_state.num_web_purchases = 2
        st.session_state.num_store_purchases = 1
        st.session_state.num_web_visits = 8
        st.session_state.recency = 90

with col3:
    if st.button("👨‍💼 Premium Customer"):
        st.session_state.age = 65
        st.session_state.income = 120000
        st.session_state.total_spending = 3500
        st.session_state.num_web_purchases = 20
        st.session_state.num_store_purchases = 15
        st.session_state.num_web_visits = 1
        st.session_state.recency = 5

st.write("### 📝 Or Enter Your Own Values:")

age = st.number_input("Age", min_value = 18, max_value = 100, 
                     value = st.session_state.get('age', 35))
income = st.number_input("Income", min_value = 0, max_value = 200000, 
                        value = st.session_state.get('income', 50000))
total_spending = st.number_input("Total Spending (sum of purchases)", min_value = 0, max_value = 5000, 
                                value = st.session_state.get('total_spending', 1000))
num_web_purchases = st.number_input("Number of Web Purchases", min_value = 0, max_value = 100, 
                                   value = st.session_state.get('num_web_purchases', 10))
num_store_purchases = st.number_input("Number of Store Purchases", min_value = 0, max_value = 100, 
                                     value = st.session_state.get('num_store_purchases', 10))

num_web_visits = st.number_input("Number of Web Visits", min_value = 0, max_value = 50, 
                                value = st.session_state.get('num_web_visits', 3))
recency = st.number_input("Rececy (days since last purchase)", min_value = 0, max_value = 365, 
                         value = st.session_state.get('recency', 30))


# Create the data with the exact column names and order expected by the scaler
data = [[age, income, total_spending, num_web_purchases, num_store_purchases, num_store_purchases, num_web_visits, recency]]
columns = ["Age", "Income", "Toatl_Spending", "NumWebPurchases", "NumStorePurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
input_data = pd.DataFrame(data, columns=columns)

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    
    # Define cluster descriptions based on the analysis
    cluster_descriptions = {
        0: "High Income, Moderate Spending Customers",
        1: "Young, Budget-Conscious Customers", 
        2: "High Income, High Spending Premium Customers",
        3: "Active, High-Value Customers",
        4: "Affluent, Store-Focused Customers",
        5: "Low Engagement, Price-Sensitive Customers"
    }
    
    st.success(f"🎯 Predicted Customer Segment: **Cluster {cluster}**")
    st.info(f"📊 Segment Type: **{cluster_descriptions.get(cluster, 'Unknown Segment')}**")
    
    # Show the input values for reference
    st.write("### Input Values Used:")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• Age: {age}")
        st.write(f"• Income: ${income:,}")
        st.write(f"• Total Spending: ${total_spending:,}")
        st.write(f"• Web Purchases: {num_web_purchases}")
    with col2:
        st.write(f"• Store Purchases: {num_store_purchases}")
        st.write(f"• Web Visits/Month: {num_web_visits}")
        st.write(f"• Days Since Last Purchase: {recency}")
    
    # Show cluster statistics
    st.write("### 💡 Try Different Values:")
    st.write("- Change the **Age** (18-100)")
    st.write("- Adjust **Income** (0-200,000)")
    st.write("- Modify **Total Spending** (0-5,000)")
    st.write("- Try different **Purchase patterns**")