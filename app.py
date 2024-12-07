from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Root route for health checks or welcome message
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Sales Analysis API. Use the /analyze endpoint to upload your dataset.'}), 200

# Favicon route
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Analyze route
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return jsonify({'message': 'Use POST to upload your dataset to /analyze.'}), 200

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded or file is invalid.'}), 400

    file = request.files['sales_data']

    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({'error': 'Failed to read the uploaded file. Please upload a valid CSV file.'}), 400

    # Validate dataset
    df.columns = df.columns.str.strip().str.lower()
    if 'month' not in df.columns or 'sales' not in df.columns:
        return jsonify({'error': 'Dataset must contain "Month" and "Sales" columns.'}), 400

    try:
        df['month'] = pd.to_datetime(df['month'])
    except Exception:
        return jsonify({'error': 'Invalid date format in "Month" column. Please ensure all dates are valid.'}), 400

    df.set_index('month', inplace=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['sales'].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Plot graph
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, y, label='Actual Sales', marker='o')
    plt.plot(df.index, y_pred, label='Predicted Sales', linestyle='--')
    plt.title('Sales Analysis')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid()

    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'plot': plot_url
    })

if __name__ == '__main__':
    # Use PORT environment variable for Render or default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
