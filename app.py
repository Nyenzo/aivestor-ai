# Importing required libraries for the Flask API and JWT authentication
from flask import Flask, request, jsonify
from stock_predictor import AdvancedStockPredictor
from chatbot import AivestorChatbot
from typing import Dict, List
import jwt
from functools import wraps
import os
from dotenv import load_dotenv
import logging
import traceback
import yfinance as yf
import time

# Setting up logging to track API requests and errors
logging.basicConfig(filename='aivestor.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Initializing the Flask app and loading environment variables
load_dotenv()
predictor = AdvancedStockPredictor()
chatbot = AivestorChatbot()
HISTORY_CACHE_TTL_SECONDS = int(os.getenv('HISTORY_CACHE_TTL_SECONDS', '300'))
history_cache = {}
SECRET_KEY = os.getenv('JWT_SECRET_KEY') or os.getenv('JWT_SECRET')
if not SECRET_KEY:
    if os.getenv('FLASK_ENV') == 'production' or os.getenv('ENV') == 'production':
        raise RuntimeError('JWT_SECRET_KEY or JWT_SECRET is required in production')
    SECRET_KEY = 'dev-only-ai-jwt-secret-change-before-prod-32'
if len(SECRET_KEY) < 32:
    if os.getenv('FLASK_ENV') == 'production' or os.getenv('ENV') == 'production':
        raise RuntimeError('AI JWT secret must be at least 32 characters in production')
    logging.warning('AI JWT secret is shorter than 32 characters; use a stronger value outside tests')
logging.info("Flask AI service initialized")

@app.after_request
def apply_cache_headers(response):
    if request.path.startswith('/predict') or request.path in ['/portfolio', '/trade_suggestions', '/history']:
        if response.status_code == 200:
            response.headers['Cache-Control'] = 'private, max-age=60, stale-while-revalidate=120'
            response.headers['X-Model-Service'] = 'aivestor-ai'
    elif request.path in ['/health', '/healthz']:
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/health', methods=['GET'])
@app.route('/healthz', methods=['GET'])
def health() -> Dict:
    model_status = predictor.get_model_status()
    return jsonify({
        'ok': True,
        'service': 'aivestor-ai',
        'model_mode': model_status['mode'],
        'model_version': 'enhanced-gradient-boosting-v2' if model_status['mode'] == 'persisted-ml' else 'technical-signal-v1'
    })

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

@app.route('/history/<ticker>', methods=['GET'])
@require_auth
def get_history(ticker: str) -> Dict:
    period = request.args.get('period', '1y')
    allowed_periods = {'5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}
    if period not in allowed_periods:
        period = '1y'

    cache_key = (ticker.upper(), period)
    cached = history_cache.get(cache_key)
    if cached and cached['expires_at'] > time.time():
        return jsonify(cached['payload'])

    try:
        history = yf.Ticker(ticker).history(period=period)
        rows = []
        if history is not None and not history.empty:
            for index, row in history.reset_index().iterrows():
                date_value = row.get('Date') or row.get('Datetime') or index
                rows.append({
                    'date': str(date_value)[:10],
                    'price': float(row.get('Close', 0) or 0),
                    'open': float(row.get('Open', 0) or 0),
                    'high': float(row.get('High', 0) or 0),
                    'low': float(row.get('Low', 0) or 0),
                    'volume': float(row.get('Volume', 0) or 0),
                })
        payload = {'ticker': ticker.upper(), 'period': period, 'data': rows}
        history_cache[cache_key] = {'expires_at': time.time() + HISTORY_CACHE_TTL_SECONDS, 'payload': payload}
        return jsonify(payload)
    except Exception as e:
        logging.warning(f"History fetch failed for {ticker}: {e}")
        return jsonify({'ticker': ticker.upper(), 'period': period, 'data': [], 'error': str(e)})

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

@app.route('/trade_suggestions', methods=['POST'])
@require_auth
def trade_suggestions() -> Dict:
    try:
        data = request.get_json() or {}
        tickers = data.get('tickers', [])
        risk_tolerance = str(data.get('risk_tolerance', 'medium')).lower()
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        suggestions = []
        for ticker in tickers[:8]:
            try:
                prediction = predictor.predict(str(ticker).upper())
                if prediction.get('error'):
                    logging.warning('Trade suggestion unavailable for %s', ticker)
                    continue
                short_signal = str(prediction.get('short_term_prediction', 'Hold'))
                long_signal = str(prediction.get('long_term_prediction', 'Hold'))
                current_price = prediction.get('current_price') or prediction.get('price') or 0
                action = 'Hold'
                if 'buy' in short_signal.lower() or 'buy' in long_signal.lower():
                    action = 'Buy'
                elif 'sell' in short_signal.lower() or 'sell' in long_signal.lower():
                    action = 'Reduce'
                confidence = prediction.get('confidence') or prediction.get('short_term_confidence') or 62
                suggestions.append({
                    'symbol': str(ticker).upper(),
                    'action': action,
                    'confidence': confidence,
                    'price': current_price,
                    'risk_tolerance': risk_tolerance,
                    'rationale': prediction.get('explanation') or f'{ticker} model signal is {short_signal} short term and {long_signal} long term.',
                })
            except Exception:
                logging.exception('Trade suggestion generation failed for %s', ticker)

        if not suggestions:
            return jsonify({'error': 'AI market model could not produce suggestions for the requested instruments'}), 503

        return jsonify({
            'model': {'name': 'Aivestor Trade Suggestions', 'version': predictor.get_model_status()['mode']},
            'suggestions': suggestions,
        })
    except Exception as e:
        logging.error(f"Trade suggestion generation failed: {str(e)}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Trade suggestions failed: {str(e)}'}), 500

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
    app.run(debug=os.getenv('FLASK_DEBUG', '').lower() == 'true', host='0.0.0.0', port=5001)
