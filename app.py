from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set logging level
logging.basicConfig(level=logging.INFO)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if file is present in request
    if 'file' not in request.files:
        app.logger.error("No 'file' key in request.files")
        return jsonify({'error': 'No file uploaded or file is invalid.'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error("File uploaded, but filename is empty")
        return jsonify({'error': 'No file uploaded or file is invalid.'}), 400

    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Validate dataset
        if 'Month' not in df.columns or 'Sales' not in df.columns:
            app.logger.error('CSV file does not contain "Month" and "Sales" columns.')
            return jsonify({'error': 'Dataset must contain "Month" and "Sales" columns.'}), 400

        # Process data
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        if df['Month'].isnull().any():
            app.logger.error("Invalid date format in 'Month' column.")
            return jsonify({'error': 'Invalid date format in "Month" column.'}), 400

        df.set_index('Month', inplace=True)
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Sales'].values

        # Train model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)

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

        app.logger.info(f"Analysis completed successfully for file: {file.filename}")
        return jsonify({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'plot': plot_url
        })

    except pd.errors.ParserError as e:
        app.logger.error(f"Error reading CSV file: {e}")
        return jsonify({'error': 'Failed to parse the CSV file. Please upload a valid CSV file.'}), 400

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Use POST to upload your dataset to /analyze.'})

if __name__ == '__main__':
    # Allow larger files for testing
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
    app.run(debug=True)
