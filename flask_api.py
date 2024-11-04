from flask import Flask, request, jsonify
import pandas as pd
import logging

app = Flask(__name__)

# Load the CSV data into a Pandas DataFrame
data = pd.read_csv('data_testing_testing.csv')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/get_house_info', methods=['GET'])
def get_house_info():
    # Get the median income parameter from the request
    median_income = request.args.get('median_income')

    if median_income is None:
        logger.error("median_income parameter is missing from the request.")
        return jsonify({'error': 'median_income parameter is required'}), 400

    try:
        median_income = float(median_income)
    except ValueError:
        logger.error(f"Invalid median_income parameter: {median_income}")
        return jsonify({'error': 'median_income parameter must be a number'}), 400

    # Filter the data by the given median income
    house_info = data[data['MEDIAN_INCOME'] == median_income]

    if house_info.empty:
        logger.info(f"No house data found for the given median_income: {median_income}")
        return jsonify({'error': 'No house data found for the given median income'}), 404

    # Convert the DataFrame to a dictionary with records format
    result = house_info.to_dict(orient='records')

    logger.info(f"Successfully retrieved house data for median_income: {median_income}")
    return jsonify(result)

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)