# Flask API for serving stock predictions, portfolio recommendations, and chatbot responses
from flask import Flask, request, jsonify
from advanced_stock_predictor import AdvancedStockPredictor
from chatbot import AivestorChatbot
from typing import Dict, List

app = Flask(__name__)
predictor = AdvancedStockPredictor()
chatbot = AivestorChatbot()

# Endpoint for predicting a single ticker
@app.route('/predict/<ticker>', methods=['GET'])
def predict_ticker(ticker: str) -> Dict:
    try:
        result = predictor.predict(ticker)
        return jsonify(result)
    except Exception as e:
        return jsonify({'ticker': ticker, 'error': f'Prediction failed: {e}'}), 500

# Endpoint for predicting multiple tickers
@app.route('/predict', methods=['POST'])
def predict_tickers() -> Dict:
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        results = predictor.predict_and_output(tickers)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

# Endpoint for predicting a sector
@app.route('/predict_sector/<sector>', methods=['GET'])
def predict_sector(sector: str) -> Dict:
    try:
        result = predictor.predict_sector(sector)
        return jsonify(result)
    except Exception as e:
        return jsonify({'sector': sector, 'error': f'Prediction failed: {e}'}), 500

# Endpoint for generating portfolio recommendations
@app.route('/portfolio', methods=['POST'])
def generate_portfolio() -> Dict:
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        risk_tolerance = data.get('risk_tolerance', 'medium')
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        result = predictor.generate_portfolio_recommendation(tickers, risk_tolerance)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Recommendation failed: {e}'}), 500

# Endpoint for chatbot FAQ responses
@app.route('/chat', methods=['POST'])
def chat() -> Dict:
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        response = chatbot.get_response(query)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Chat failed: {e}'}), 500

if __name__ == "__main__":
    # Running the Flask API
    app.run(debug=True, host='0.0.0.0', port=5000)