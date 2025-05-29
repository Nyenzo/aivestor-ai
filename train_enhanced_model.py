from advanced_stock_predictor import AdvancedStockPredictor, save_models, load_models
import json
import pandas as pd
from datetime import datetime

def load_sentiment_data():
    """Load and process sentiment data from sentiment_results.txt"""
    sentiment_results = {}
    current_section = ""
    
    with open("sentiment_results.txt", 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() in ["USA News Sentiments:"]:
                current_section = "USA"
            elif "Sentiment:" in line and "Confidence:" in line:
                _, sentiment_conf = line.split(" | Sentiment: ")
                sentiment, confidence = sentiment_conf.split(" | Confidence: ")
                confidence = float(confidence.strip())
                sentiment_score = 1.0 if sentiment == "POSITIVE" else -1.0 if sentiment == "NEGATIVE" else 0.0
                
                if current_section == "USA":
                    # Apply USA sentiment to all sectors
                    for sector in ['Technology', 'Healthcare', 'Financials', 
                                 'Consumer Discretionary', 'Consumer Staples',
                                 'Energy', 'Industrials', 'Utilities']:
                        if sector not in sentiment_results:
                            sentiment_results[sector] = []
                        sentiment_results[sector].append({
                            'sentiment': sentiment_score,
                            'confidence': confidence
                        })
    
    # Average sentiment scores for each sector
    final_sentiments = {}
    for sector, scores in sentiment_results.items():
        if scores:
            avg_sentiment = sum(score['sentiment'] * score['confidence'] for score in scores) / \
                          sum(score['confidence'] for score in scores)
            final_sentiments[sector] = {'sentiment': avg_sentiment}
    
    return final_sentiments

def main():
    print("Starting enhanced model training process...")
    
    # Initialize predictor
    predictor = AdvancedStockPredictor()
    
    # Load sentiment data
    print("\nLoading sentiment data...")
    sentiment_results = load_sentiment_data()
    print(f"Loaded sentiment data for {len(sentiment_results)} sectors")
    
    # Train models
    print("\nTraining models...")
    predictor.train_models(sector_data=None, sentiment_results=sentiment_results)
    
    # Save trained models
    print("\nSaving trained models...")
    save_models(predictor)
    
    # Make predictions for each sector
    print("\nGenerating predictions for each sector...")
    predictions = {}
    for sector, ticker in predictor.sector_etfs.items():
        sentiment_score = sentiment_results.get(sector, {'sentiment': 0.0})['sentiment']
        pred = predictor.predict_sector(sector, ticker, sentiment_score)
        if pred:
            predictions[sector] = pred
            print(f"\n{sector} Predictions:")
            print(f"Short-term: {pred['short_term']['prediction']} (Confidence: {pred['short_term']['confidence']:.2f})")
            print(f"Long-term: {pred['long_term']['prediction']} (Confidence: {pred['long_term']['confidence']:.2f})")
    
    # Generate portfolio recommendations
    print("\nGenerating portfolio recommendations...")
    for risk_level in ['conservative', 'moderate', 'aggressive']:
        print(f"\n{risk_level.capitalize()} Portfolio Recommendations:")
        recommendations = predictor.generate_portfolio_recommendation(predictions, risk_level)
        
        print("\nTop Sectors to Invest:")
        for sector in recommendations['top_sectors']:
            print(f"- {sector['sector']} (ETF: {sector['etf']}, Score: {sector['score']:.2f})")
        
        print("\nSectors to Avoid:")
        for sector in recommendations['avoid_sectors']:
            print(f"- {sector['sector']} (ETF: {sector['etf']}, Score: {sector['score']:.2f})")
    
    # Save predictions and recommendations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'predictions': predictions,
        'recommendations': {
            'conservative': predictor.generate_portfolio_recommendation(predictions, 'conservative'),
            'moderate': predictor.generate_portfolio_recommendation(predictions, 'moderate'),
            'aggressive': predictor.generate_portfolio_recommendation(predictions, 'aggressive')
        }
    }
    
    with open(f'prediction_results_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"\nResults saved to prediction_results_{timestamp}.json")

if __name__ == "__main__":
    main() 