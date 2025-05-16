---
title: SuperKart Sales Predictor
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# SuperKart Sales Predictor API

This API predicts the sales of products in SuperKart stores based on various product and store attributes.

## API Endpoints

### GET /
Returns a welcome message.

### POST /predict
Predicts the sales for a product in a store.

#### Request Format
```json
{
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
}
```

#### Response Format
```json
{
  "status": "success",
  "Predicted_Sales": 2865.41
}
```

### GET /test-model
Tests the model with a sample input and returns the prediction.

## Model Information
This API uses an XGBoost model trained on SuperKart sales data. The model predicts the sales of a product in a store based on various attributes.
