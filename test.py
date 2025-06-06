# Validating prediction accuracy with real historical stock data
import pandas as pd
import numpy as np
from advanced_stock_predictor import AdvancedStockPredictor
from sklearn.metrics import accuracy_score
from typing import List, Dict
import yfinance as yf

class PortfolioTester:
    # Initializing tester with predictor
    def __init__(self):
        self.predictor = AdvancedStockPredictor()
        try:
            self.predictor.load_models()
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            raise

    # Generating real historical data for validation
    def generate_real_data(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        data = {}
        for ticker in tickers:
            df = self.predictor._fetch_historical_data(ticker, days + 63)  # Extra days for indicators
            if df.empty:
                continue
            df = self.predictor._compute_indicators(df)
            # Define actual signals based on next day's close movement
            df['Actual_Short_Signal'] = np.where(df['Close'].pct_change().shift(-1) > 0, 'Buy',
                                               np.where(df['Close'].pct_change().shift(-1) < 0, 'Sell', 'Hold'))
            df['Actual_Long_Signal'] = np.where(df['Close'].rolling(63).mean().pct_change().shift(-63) > 0, 'Buy',
                                              np.where(df['Close'].rolling(63).mean().pct_change().shift(-63) < 0, 'Sell', 'Hold'))
            data[ticker] = df.iloc[:-63]  # Exclude last 63 days to align predictions
        return data

    # Fetching historical data for a ticker
    def _fetch_historical_data(self, ticker: str, days: int) -> pd.DataFrame:
        try:
            df = yf.Ticker(ticker).history(period=f'{days}d')
            return df[['Open', 'High', 'Low', 'Close']] if not df.empty else pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    # Testing prediction accuracy with real data
    def test_accuracy(self, tickers: List[str], days: int = 252) -> Dict:
        try:
            real_data = self.generate_real_data(tickers, days)
            results = {'short_term': {}, 'long_term': {}}

            for ticker in tickers:
                if ticker not in real_data or real_data[ticker].empty:
                    continue
                df = real_data[ticker]
                short_true = []
                short_pred = []
                long_true = []
                long_pred = []

                for i in range(len(df) - 1):  # Exclude last day for next-day comparison
                    latest_data = {
                        'Close': df['Close'].iloc[i],
                        'RSI': df['RSI'].iloc[i],
                        'MACD': df['MACD'].iloc[i],
                        'BB_upper': df['BB_upper'].iloc[i],
                        'BB_lower': df['BB_lower'].iloc[i],
                        'ATR': df['ATR'].iloc[i],
                        'VIX': self.predictor._fetch_vix(),
                        'Sector_Sentiment': self.predictor._fetch_sector_sentiment(self.predictor.get_sector(ticker)),
                        **self.predictor._fetch_economic_data()
                    }
                    features = pd.DataFrame([latest_data])
                    features_scaled = self.predictor.scaler.transform(features.values)

                    short_pred_val = self.predictor.short_term_model.predict(features_scaled)[0]
                    long_pred_val = self.predictor.long_term_model.predict(features_scaled)[0]
                    signal_map = {1: 'Buy', 0: 'Sell', 2: 'Hold'}

                    short_true.append(df['Actual_Short_Signal'].iloc[i])
                    short_pred.append(signal_map[short_pred_val])
                    long_true.append(df['Actual_Long_Signal'].iloc[i])
                    long_pred.append(signal_map[long_pred_val])

                results['short_term'][ticker] = accuracy_score(short_true, short_pred) if short_true else 0.0
                results['long_term'][ticker] = accuracy_score(long_true, long_pred) if long_true else 0.0

            print("\nAccuracy Results:")
            for ticker in tickers:
                if ticker in results['short_term']:
                    print(f"{ticker} Short-Term Accuracy: {results['short_term'][ticker]:.4f}")
                    print(f"{ticker} Long-Term Accuracy: {results['long_term'][ticker]:.4f}")

            results['summary'] = {
                'short_term_avg': np.mean(list(results['short_term'].values())),
                'long_term_avg': np.mean(list(results['long_term'].values()))
            }
            print(f"\nSummary: Short-Term Avg Accuracy = {results['summary']['short_term_avg']:.4f}, Long-Term Avg Accuracy = {results['summary']['long_term_avg']:.4f}")
            return results
        except Exception as e:
            print(f"Error testing accuracy: {e}")
            return {}

if __name__ == "__main__":
    # Executing accuracy test with real data for sample portfolio
    tester = PortfolioTester()
    tester.test_accuracy(['AAPL', 'XRT', 'LIN'])