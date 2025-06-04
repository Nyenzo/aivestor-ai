import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from datetime import datetime
import numpy as np

class MarketDataAnalyzer:
    def __init__(self):
        self.stock_data = {}
        self.economic_data = None
        self.market_sentiment = None
        
    def load_data(self):
        # Load stock data from CSV files
        for file in glob.glob('stock_data_*.csv'):
            ticker = file.split('_')[2]
            self.stock_data[ticker] = pd.read_csv(file, index_col=0, parse_dates=True)
            
        # Load economic indicators
        economic_files = glob.glob('economic_indicators_*.csv')
        if economic_files:
            latest_file = max(economic_files, key=os.path.getctime)
            self.economic_data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
        # Load market sentiment
        if os.path.exists('market_data.json'):
            with open('market_data.json', 'r') as f:
                sentiment_data = json.load(f)
                self.market_sentiment = sentiment_data.get('market_sentiment', {})
    
    def plot_sector_performance(self):
        plt.figure(figsize=(15, 8))
        for ticker, data in self.stock_data.items():
            if len(data) > 0:
                returns = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                plt.plot(returns.index, returns, label=ticker)
        
        plt.title('Sector Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('sector_performance.png')
        plt.close()
    
    def plot_volatility_heatmap(self):
        volatilities = {}
        for ticker, data in self.stock_data.items():
            if len(data) > 0:
                volatilities[ticker] = data['Volatility'].mean() * 100
        
        if volatilities:
            plt.figure(figsize=(12, 6))
            volatilities_series = pd.Series(volatilities)
            sns.barplot(x=volatilities_series.index, y=volatilities_series.values)
            plt.title('Average Volatility by Sector')
            plt.xlabel('Ticker')
            plt.ylabel('Average Volatility (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('volatility_comparison.png')
            plt.close()
    
    def plot_economic_indicators(self):
        if self.economic_data is not None:
            # Plot key economic indicators
            indicators = ['GDP', 'Inflation', 'Unemployment', 'Fed_Funds_Rate']
            fig, axes = plt.subplots(len(indicators), 1, figsize=(15, 4*len(indicators)))
            
            for i, indicator in enumerate(indicators):
                if indicator in self.economic_data.columns:
                    data = self.economic_data[indicator].dropna()
                    axes[i].plot(data.index, data.values)
                    axes[i].set_title(f'{indicator} Over Time')
                    axes[i].grid(True)
            
            plt.tight_layout()
            plt.savefig('economic_indicators.png')
            plt.close()
    
    def plot_sentiment_analysis(self):
        if self.market_sentiment:
            plt.figure(figsize=(12, 6))
            sectors = list(self.market_sentiment.keys())
            sentiments = list(self.market_sentiment.values())
            
            colors = ['g' if s > 0 else 'r' for s in sentiments]
            plt.bar(sectors, sentiments, color=colors)
            plt.title('Market Sentiment by Sector')
            plt.xlabel('Sector')
            plt.ylabel('Sentiment Score')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig('sentiment_analysis.png')
            plt.close()
    
    def generate_correlation_matrix(self):
        # Create correlation matrix for stock returns
        returns_dict = {}
        for ticker, data in self.stock_data.items():
            if len(data) > 0:
                returns_dict[ticker] = data['Close'].pct_change()
        
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            correlation = returns_df.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('Cross-Sector Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png')
            plt.close()
    
    def analyze_all(self):
        print("Loading data...")
        self.load_data()
        
        print("\nGenerating visualizations...")
        self.plot_sector_performance()
        self.plot_volatility_heatmap()
        self.plot_economic_indicators()
        self.plot_sentiment_analysis()
        self.generate_correlation_matrix()
        
        print("\nAnalysis complete! Generated visualization files:")
        print("- sector_performance.png")
        print("- volatility_comparison.png")
        print("- economic_indicators.png")
        print("- sentiment_analysis.png")
        print("- correlation_matrix.png")

if __name__ == "__main__":
    analyzer = MarketDataAnalyzer()
    analyzer.analyze_all() 