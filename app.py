# Importing required libraries for the Flask API and JWT authentication
from flask import Flask, request, jsonify
from advanced_stock_predictor import AdvancedStockPredictor
from chatbot import AivestorChatbot
from typing import Dict, List
import jwt
from functools import wraps
import os
from dotenv import load_dotenv
import logging
import traceback

# Setting up logging to track API requests and errors
logging.basicConfig(filename='aivestor.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Initializing the Flask app and loading environment variables
load_dotenv()
predictor = AdvancedStockPredictor()
chatbot = AivestorChatbot()
SECRET_KEY = os.getenv('JWT_SECRET_KEY') or 'your-very-secure-secret-key'
logging.info("Flask AI service initialized")

# Middleware to verify JWT tokens
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            logging.warning("Token missing in request")
            return jsonify({'error': 'Token is missing'}), 401
        try:
            jwt.decode(token.replace('Bearer ', ''), SECRET_KEY, algorithms=['HS256'])
            logging.info("JWT token verified")
        except jwt.InvalidTokenError as e:
            logging.warning(f"Invalid JWT token: {str(e)}")
            return jsonify({'error': f'Invalid token: {str(e)}'}), 401
        return f(*args, **kwargs)
    return decorated

# Endpoint for predicting a single ticker
@app.route('/predict/<ticker>', methods=['GET'])
@require_auth
def predict_ticker(ticker: str) -> Dict:
    try:
        logging.debug(f"Starting prediction for ticker: {ticker}")
        result = predictor.predict(ticker)
        logging.info(f"Prediction successful for {ticker}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Prediction failed for {ticker}: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({'ticker': ticker, 'error': f'Prediction failed: {str(e)}'}), 500

# Endpoint for predicting multiple tickers
@app.route('/predict', methods=['POST'])
@require_auth
def predict_tickers() -> Dict:
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        logging.debug(f"Starting predictions for tickers: {tickers}")
        if not tickers:
            logging.warning("No tickers provided in request")
            return jsonify({'error': 'No tickers provided'}), 400
        results = predictor.predict_and_output(tickers)
        logging.info(f"Predictions successful for {tickers}")
        return jsonify(results)
    except Exception as e:
        logging.error(f"Multiple ticker prediction failed: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Endpoint for predicting a sector
@app.route('/predict_sector/<sector>', methods=['GET'])
@require_auth
def predict_sector(sector: str) -> Dict:
    try:
        logging.debug(f"Starting sector prediction for: {sector}")
        result = predictor.predict_sector(sector)
        logging.info(f"Sector prediction successful for {sector}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Sector prediction failed for {sector}: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({'sector': sector, 'error': f'Prediction failed: {str(e)}'}), 500

# Endpoint for generating portfolio recommendations
@app.route('/portfolio', methods=['POST'])
@require_auth
def generate_portfolio() -> Dict:
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        risk_tolerance = data.get('risk_tolerance', 'medium')
        logging.debug(f"Starting portfolio recommendation for tickers: {tickers}, risk: {risk_tolerance}")
        if not tickers:
            logging.warning("No tickers provided for portfolio")
            return jsonify({'error': 'No tickers provided'}), 400
        result = predictor.generate_portfolio_recommendation(tickers, risk_tolerance)
        logging.info(f"Portfolio recommendation successful for {tickers}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Portfolio recommendation failed: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Recommendation failed: {str(e)}'}), 500

# Endpoint for chatbot FAQ responses
@app.route('/chat', methods=['POST'])
@require_auth
def chat() -> Dict:
    try:
        data = request.get_json()
        query = data.get('query', '')
        logging.debug(f"Starting chatbot query: {query}")
        if not query:
            logging.warning("No query provided for chatbot")
            return jsonify({'error': 'No query provided'}), 400
        response = chatbot.get_response(query)
        logging.info(f"Chatbot query successful: {query}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Chatbot query failed: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

# Running the Flask API on port 5001
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)