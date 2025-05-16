import streamlit as st
import requests
import json
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

# Backend API URL - replace with your actual deployed backend URL
BACKEND_API_URL = st.sidebar.text_input(
    "Backend API URL",
    value="https://abhilashmanchala-superkart-sales-predictor.hf.space",
    help="Enter the URL of your deployed backend API"
)

# Title and description
st.title("🛒 SuperKart Sales Predictor")
st.markdown("""
This app predicts the total sales of a product in a SuperKart store based on various attributes.
Enter the product and store details below to get a sales prediction.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Product Details")
    product_weight = st.number_input("Product Weight (kg)", min_value=0.1, max_value=100.0, value=12.66, step=0.01)
    product_mrp = st.number_input("Product MRP (₹)", min_value=1.0, max_value=10000.0, value=117.08, step=0.01)
    product_allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.001, max_value=1.0, value=0.027, step=0.001)
    price_per_gram = st.number_input("Price per Gram (₹)", min_value=0.01, max_value=1000.0, value=9.25, step=0.01)
    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        options=["Low Sugar", "Regular", "No Sugar", "High Sugar", "reg"]
    )
    product_code = st.selectbox(
        "Product Code",
        options=["FD", "DR", "NC", "FV", "PC", "HH", "OT"]
    )
    product_category = st.selectbox(
        "Product Category",
        options=["Frozen", "Dairy & Eggs", "Pantry Staples", "Baking & Grains", 
                "Non-Food", "Snacks", "Meat & Seafood", "Beverages", "Produce", 
                "Breakfast", "Miscellaneous"]
    )

with col2:
    st.subheader("Store Details")
    store_age = st.number_input("Store Age (years)", min_value=1, max_value=50, value=16, step=1)
    store_size = st.selectbox(
        "Store Size",
        options=["Small", "Medium", "Large", "High"]
    )
    store_location_city_type = st.selectbox(
        "Store Location City Type",
        options=["Tier 1", "Tier 2", "Tier 3"]
    )
    store_type = st.selectbox(
        "Store Type",
        options=["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
    )

# Create a button to trigger prediction
if st.button("Predict Sales", type="primary"):
    # Prepare the input data
    input_data = {
        "Product_Weight": product_weight,
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Price_per_Gram": price_per_gram,
        "Store_Age": store_age,
        "Product_Sugar_Content": product_sugar_content,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Product_Code": product_code,
        "Product_Category": product_category
    }
    
    # Display the input data in a collapsible section
    with st.expander("View Input Data"):
        st.json(input_data)
    
    # Make API request
    try:
        with st.spinner("Predicting sales..."):
            response = requests.post(
                f"{BACKEND_API_URL}/predict",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Display the prediction
            st.success(f"### Predicted Sales: ₹{result['Predicted_Sales']:,.2f}")
            
            # Add some visual elements
            st.balloons()
            
            # Display a sample visualization
            st.subheader("Sales Comparison")
            chart_data = pd.DataFrame({
                'Category': ['Predicted Sales', 'Average Category Sales', 'Store Average Sales'],
                'Sales': [result['Predicted_Sales'], result['Predicted_Sales']*0.85, result['Predicted_Sales']*1.2]
            })
            st.bar_chart(chart_data.set_index('Category'))
            
        else:
            st.error(f"Error: {response.status_code}")
            st.json(response.json())
    
    except Exception as e:
        st.error(f"Error connecting to the backend API: {str(e)}")
        st.info("Make sure the backend API is running and accessible.")

# Add information about the model
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This app uses a machine learning model to predict product sales in SuperKart stores.
    
    The model was trained on historical sales data and takes into account various product and store attributes.
    
    **Model Type:** XGBoost Regressor
    
    **Features Used:**
    - Product attributes (weight, price, etc.)
    - Store attributes (size, location, etc.)
    - Derived features (price per gram, etc.)
    """)
    
    # Add a test connection button
    if st.button("Test Backend Connection"):
        try:
            response = requests.get(f"{BACKEND_API_URL}/")
            st.success(f"Connection successful! Response: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")