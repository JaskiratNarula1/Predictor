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

        # Line Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, y, label='Actual Sales', marker='o')
        plt.plot(df.index, y_pred, label='Predicted Sales', linestyle='--')
        plt.title('Sales Analysis')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid()

        # Save line plot to base64 string
        line_img = io.BytesIO()
        plt.savefig(line_img, format='png')
        line_img.seek(0)
        line_plot_url = base64.b64encode(line_img.getvalue()).decode()
        plt.close()

        # Bar Graph
        plt.figure(figsize=(10, 6))
        width = 0.4
        months = df.index.strftime('%Y-%m')  # Format month for bar labels
        indices = np.arange(len(df))

        plt.bar(indices - width/2, y, width=width, label='Actual Sales')
        plt.bar(indices + width/2, y_pred, width=width, label='Predicted Sales')
        plt.xticks(indices, months, rotation=45)
        plt.title('Sales Comparison')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.legend()
        plt.tight_layout()

        # Save bar graph to base64 string
        bar_img = io.BytesIO()
        plt.savefig(bar_img, format='png')
        bar_img.seek(0)
        bar_plot_url = base64.b64encode(bar_img.getvalue()).decode()
        plt.close()

        app.logger.info(f"Analysis completed successfully for file: {file.filename}")
        return jsonify({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'line_plot': line_plot_url,
            'bar_plot': bar_plot_url
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
