import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
superkart_sales_predictor_api = Flask(__name__)
CORS(superkart_sales_predictor_api)

# Load the model
try:
    model = joblib.load("superkart_sales_prediction_model_v1_0.joblib")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature engineering functions
def create_product_code(df):
    df['Product_Code'] = df['Product_Id'].str[:2] if 'Product_Id' in df.columns else df['Product_Code']
    return df

@superkart_sales_predictor_api.route('/')
def home():
    return "Welcome to the SuperKart Sales Predictor API"

@superkart_sales_predictor_api.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return prediction
        return jsonify({
            "status": "success",
            "Predicted_Sales": float(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@superkart_sales_predictor_api.route('/test-model', methods=['GET'])
def test_model():
    try:
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
        
        # Model info to return
        model_info = {
            "model_loaded": model is not None,
            "sample_input": sample_input.to_dict(orient="records")[0]
        }
        
        # Try to make a prediction
        if model is not None:
            prediction = model.predict(sample_input)[0]
            model_info["test_prediction"] = float(prediction)
        
        return jsonify(model_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    superkart_sales_predictor_api.run(debug=False, host='0.0.0.0', port=7860)

