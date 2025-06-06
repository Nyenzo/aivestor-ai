# Processing collected data into features and trading signals for model training
import pandas as pd
import pickle
import os
from datetime import datetime
from typing import Dict, List
from advanced_stock_predictor import AdvancedStockPredictor

class EnhancedDataProcessor:
    DATA_DIR = 'datacollection'

    # Initializing the data processor with collected data path
    def __init__(self, collected_data_path: str):
        self.collected_data_path = collected_data_path
        self.data = self.load_data()
        self.predictor = AdvancedStockPredictor()

    # Loading collected data from pickle file
    def load_data(self) -> Dict:
        try:
            with open(self.collected_data_path, 'rb') as f:
                data = pickle.load(f)
            print("Loaded collected data")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}

    # Processing data into features for all tickers
    def process_features(self) -> Dict[str, pd.DataFrame]:
        print("Starting feature processing")
        stock_data = self.data.get('stock_data', {})
        economic_data = self.data.get('economic_data', pd.DataFrame())
        vix_data = self.data.get('vix_data', pd.DataFrame())
        fundamental_data = self.data.get('fundamental_data', {})
        market_sentiment = self.data.get('market_sentiment', {})
        processed_features = {}

        for ticker in stock_data.keys():
            try:
                df = stock_data[ticker].copy()
                if df.empty or 'Close' not in df.columns:
                    print(f"Skipping {ticker}: empty DataFrame or missing 'Close' column (shape: {df.shape}, columns: {df.columns.tolist()})")
                    continue

                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df = df.sort_values('Date')

                # Merging with VIX data
                if not vix_data.empty and 'VIX' in vix_data.columns:
                    vix_data['Date'] = pd.to_datetime(vix_data['Date'], utc=True)
                    df = df.merge(vix_data[['Date', 'VIX']], on='Date', how='left')

                # Merging with economic data
                if not economic_data.empty:
                    economic_data['Date'] = pd.to_datetime(economic_data['Date'], utc=True)
                    df = df.merge(economic_data, on='Date', how='left')

                # Adding fundamental data
                if ticker in fundamental_data:
                    for key, value in fundamental_data[ticker].items():
                        df[key] = value

                # Adding sector sentiment
                sector = self.predictor.get_sector(ticker)
                if sector in market_sentiment and sector != 'Unknown':
                    sentiment_scores = [s['score'] for s in market_sentiment[sector] if s.get('sentiment') == 'POSITIVE']
                    df['Sector_Sentiment'] = sum(sentiment_scores) / max(len(sentiment_scores), 1) if sentiment_scores else 0.0
                else:
                    df['Sector_Sentiment'] = 0.5

                # Filling missing values
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    df[col] = df[col].ffill().bfill().fillna(0.0)

                # Generating trading signals
                df['Short_Term_Signal'] = self.generate_signal(df['Close'], window=63, threshold=0.05)
                df['Long_Term_Signal'] = self.generate_signal(df['Close'], window=252, threshold=0.15)

                processed_features[ticker] = df
                print(f"Processed {ticker}: {df.shape[0]} rows, columns: {list(df.columns)}")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        print(f"Processed features for {len(processed_features)} tickers")
        return processed_features

    # Generating trading signals based on price changes
    def generate_signal(self, prices: pd.Series, window: int, threshold: float) -> pd.Series:
        try:
            returns = prices.pct_change(periods=window)
            signals = pd.Series('Hold', index=prices.index)
            signals[returns > threshold] = 'Buy'
            signals[returns < -threshold] = 'Sell'
            return signals
        except Exception as e:
            print(f"Error generating signals: {e}")
            return pd.Series('Hold', index=prices.index)

    # Saving processed features to pickle file
    def save_features(self, features: Dict[str, pd.DataFrame]):
        try:
            output_path = os.path.join(self.DATA_DIR, 'processed_features.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Saved processed features to '{output_path}'")
        except Exception as e:
            print(f"Error saving features: {e}")

if __name__ == "__main__":
    # Executing feature processing and saving results
    processor = EnhancedDataProcessor(os.path.join('datacollection', 'collected_data.pkl'))
    features = processor.process_features()
    processor.save_features(features)