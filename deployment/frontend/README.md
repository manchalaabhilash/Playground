---
title: SuperKart Sales Predictor UI
emoji: 🛒
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# SuperKart Sales Predictor - Frontend

This is a Streamlit frontend for the SuperKart Sales Predictor API. It provides a user-friendly interface for predicting product sales in SuperKart stores.

## Features

- Input form for product and store attributes
- Real-time sales prediction
- Visualization of prediction results
- Connection to the backend API

## How to Use

1. Enter the product details (weight, price, etc.)
2. Enter the store details (size, location, etc.)
3. Click "Predict Sales" to get a prediction
4. View the results and visualizations

## Backend API

This frontend connects to a backend API that hosts the machine learning model. The backend API is available at:
https://huggingface.co/spaces/YOUR_USERNAME/superkart-sales-predictor

## Technologies Used

- Streamlit
- Python
- Pandas
- Requests (for API calls)
