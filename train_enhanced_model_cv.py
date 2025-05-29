from advanced_stock_predictor import AdvancedStockPredictor, save_models, load_models
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate various evaluation metrics"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'prediction_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def plot_feature_importance(feature_importance: dict):
    """Plot feature importance"""
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    sns.barplot(data=importance_df, y='Feature', x='Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def main():
    print("Starting enhanced model training process with cross-validation...")
    
    # Initialize predictor
    predictor = AdvancedStockPredictor()
    
    # Load sentiment data
    print("\nLoading sentiment data...")
    sentiment_results = load_sentiment_data()
    print(f"Loaded sentiment data for {len(sentiment_results)} sectors")
    
    # Collect enhanced data
    print("\nCollecting enhanced historical and fundamental data...")
    collected_data = predictor.data_collector.collect_all_data(
        predictor.sector_etfs.values(), 
        start_date='2018-01-01'
    )
    
    # Prepare data for cross-validation
    X_all = []
    y_short_all = []
    y_long_all = []
    
    print("\nPreparing data for cross-validation...")
    for sector, ticker in predictor.sector_etfs.items():
        stock_data = collected_data['stock_data'].get(ticker)
        if stock_data is None:
            continue
            
        sector_sentiment = sentiment_results.get(sector, {'sentiment': 0.0})['sentiment']
        
        # Prepare features with fundamental data
        features = predictor.prepare_features(
            stock_data=stock_data,
            economic_data=collected_data['economic_indicators'],
            sentiment_score=sector_sentiment
        )
        
        # Create labels
        short_term_returns = stock_data['Close'].pct_change(periods=63)  # ~3 months
        long_term_returns = stock_data['Close'].pct_change(periods=252)  # ~1 year
        
        # Remove rows with NaN values
        valid_idx = ~(features.isna().any(axis=1) | short_term_returns.isna() | long_term_returns.isna())
        
        if valid_idx.sum() > 0:
            X_all.extend(features[valid_idx].values)
            y_short_all.extend((short_term_returns[valid_idx] > 0).astype(int))
            y_long_all.extend((long_term_returns[valid_idx] > 0).astype(int))
    
    X_all = np.array(X_all)
    y_short_all = np.array(y_short_all)
    y_long_all = np.array(y_long_all)
    
    # Initialize TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Metrics storage
    short_term_metrics = []
    long_term_metrics = []
    
    print("\nPerforming cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        print(f"\nFold {fold + 1}/5")
        
        # Split data
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_short_train, y_short_test = y_short_all[train_idx], y_short_all[test_idx]
        y_long_train, y_long_test = y_long_all[train_idx], y_long_all[test_idx]
        
        # Scale features
        X_train_scaled = predictor.scaler.fit_transform(X_train)
        X_test_scaled = predictor.scaler.transform(X_test)
        
        # Train and evaluate short-term model
        predictor.short_term_model.fit(X_train_scaled, y_short_train)
        y_short_pred = predictor.short_term_model.predict(X_test_scaled)
        short_term_metrics.append(evaluate_predictions(y_short_test, y_short_pred))
        
        # Train and evaluate long-term model
        predictor.long_term_model.fit(X_train_scaled, y_long_train)
        y_long_pred = predictor.long_term_model.predict(X_test_scaled)
        long_term_metrics.append(evaluate_predictions(y_long_test, y_long_pred))
    
    # Plot cross-validation results
    plot_predictions(y_short_all, y_short_pred, 'Short-term Model Predictions')
    plot_predictions(y_long_all, y_long_pred, 'Long-term Model Predictions')
    
    # Train final models on full dataset
    print("\nTraining final models on full dataset...")
    X_scaled = predictor.scaler.fit_transform(X_all)
    predictor.short_term_model.fit(X_scaled, y_short_all)
    predictor.long_term_model.fit(X_scaled, y_long_all)
    
    # Save trained models
    print("\nSaving trained models...")
    save_models(predictor)
    
    # Generate and save predictions
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
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'predictions': predictions,
        'cross_validation_metrics': {
            'short_term': short_term_metrics,
            'long_term': long_term_metrics
        },
        'recommendations': {
            'conservative': predictor.generate_portfolio_recommendation(predictions, 'conservative'),
            'moderate': predictor.generate_portfolio_recommendation(predictions, 'moderate'),
            'aggressive': predictor.generate_portfolio_recommendation(predictions, 'aggressive')
        }
    }
    
    with open(f'prediction_results_cv_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"\nResults saved to prediction_results_cv_{timestamp}.json")
    print("\nCross-validation plots saved as Short-term_Model_cv_metrics.png and Long-term_Model_cv_metrics.png")

    # Plot feature importance
    plot_feature_importance(predictor.feature_importance)

if __name__ == "__main__":
    main() 