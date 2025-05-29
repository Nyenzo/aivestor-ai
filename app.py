from flask import Flask, request, jsonify
from dotenv import load_dotenv
from advanced_stock_predictor import AdvancedStockPredictor
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize AI model
predictor = AdvancedStockPredictor()

@app.route('/')
def hello():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Aivestor AI API',
        'version': '1.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate stock predictions and recommendations"""
    try:
        data = request.get_json()
        sector = data.get('sector')
        ticker = data.get('ticker')
        sentiment_score = data.get('sentiment_score', 0.0)

        # Get predictions
        prediction = predictor.predict_sector(
            sector=sector,
            ticker=ticker,
            sentiment_score=sentiment_score
        )

        return jsonify({
            'status': 'success',
            'data': prediction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/portfolio/recommend', methods=['POST'])
def recommend_portfolio():
    """Generate portfolio recommendations"""
    try:
        data = request.get_json()
        predictions = data.get('predictions', {})
        risk_tolerance = data.get('risk_tolerance', 'moderate')

        # Generate recommendations
        recommendations = predictor.generate_portfolio_recommendation(
            predictions=predictions,
            risk_tolerance=risk_tolerance
        )

        return jsonify({
            'status': 'success',
            'data': recommendations
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Train or update the AI model"""
    try:
        data = request.get_json()
        sector_data = data.get('sector_data', {})
        sentiment_results = data.get('sentiment_results', {})

        # Train models
        predictor.train_models(
            sector_data=sector_data,
            sentiment_results=sentiment_results
        )

        return jsonify({
            'status': 'success',
            'message': 'Model training completed successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure all environment variables are loaded
    required_vars = ['ALPHA_VANTAGE_API_KEY', 'FRED_API_KEY', 'NEWS_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    app.run(host='0.0.0.0', port=5001, debug=True)