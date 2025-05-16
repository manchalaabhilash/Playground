import joblib
import pandas as pd

# Test loading the model
try:
    print("Loading model...")
    model = joblib.load("superkart_sales_prediction_model_v1_0.joblib")
    print("Model loaded successfully!")
    
    # Create a sample input
    sample_input = pd.DataFrame([{
        "Product_Weight": 12.66,
        "Product_Allocated_Area": 0.027,
        "Product_MRP": 117.08,
        "Price_per_Gram": 9.2480252764613,
        "Store_Age": 16,
        "Product_Sugar_Content": "Low Sugar",
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2",
        "Product_Code": "FD",
        "Product_Category": "Frozen"
    }])
    
    # Make a prediction
    print("Making prediction...")
    prediction = model.predict(sample_input)[0]
    print(f"Prediction successful! Predicted sales: {prediction}")
    
except Exception as e:
    print(f"Error: {e}")