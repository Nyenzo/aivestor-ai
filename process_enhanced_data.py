import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any
import os

class DataProcessor:
    def __init__(self):
        self.market_data = None
        self.economic_data = None
        self.fundamental_data = None
        
    def load_data(self):
        """Load all collected data"""
        try:
            # Load market data
            with open('market_data.json', 'r') as f:
                market_data = json.load(f)
                self.market_data = {k: pd.DataFrame.from_dict(v, orient='split') 
                                  for k, v in market_data.items()}
            
            # Load economic data
            with open('economic_data.json', 'r') as f:
                self.economic_data = json.load(f)
                
            # Load fundamental data
            with open('fundamental_data.json', 'r') as f:
                self.fundamental_data = json.load(f)
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
        return True
        
    def process_market_data(self):
        """Process market data and calculate additional features"""
        processed_data = {}
        
        for sector, df in self.market_data.items():
            # Calculate momentum indicators
            df['Momentum_1M'] = df['Close'].pct_change(periods=21)  # 21 trading days
            df['Momentum_3M'] = df['Close'].pct_change(periods=63)  # 63 trading days
            df['Momentum_6M'] = df['Close'].pct_change(periods=126)  # 126 trading days
            
            # Calculate volume indicators
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            # Calculate price channels
            df['High_20d'] = df['High'].rolling(window=20).max()
            df['Low_20d'] = df['Low'].rolling(window=20).min()
            df['Channel_Position'] = (df['Close'] - df['Low_20d']) / (df['High_20d'] - df['Low_20d'])
            
            processed_data[sector] = df
            
        return processed_data
        
    def process_economic_data(self):
        """Process economic indicators and calculate trends"""
        processed_econ = {}
        
        for indicator, data in self.economic_data.items():
            if isinstance(data, dict):  # Convert dictionary to Series if needed
                series = pd.Series(data)
                series.index = pd.to_datetime(series.index)
            else:
                series = pd.Series(data)
                
            # Calculate YoY change
            processed_econ[f"{indicator}_YoY"] = series.pct_change(periods=12)
            
            # Calculate 3-month trend
            processed_econ[f"{indicator}_3M_Trend"] = series.rolling(window=3).mean()
            
        return processed_econ
        
    def process_fundamental_data(self):
        """Process fundamental data and calculate sector averages"""
        processed_fundamentals = {}
        
        for sector, companies in self.fundamental_data.items():
            sector_metrics = {
                'Avg_PE': [],
                'Avg_PB': [],
                'Avg_Dividend_Yield': [],
                'Total_Market_Cap': [],
                'Avg_Revenue_Growth': []
            }
            
            for company, metrics in companies.items():
                if metrics['PE_Ratio']:
                    sector_metrics['Avg_PE'].append(metrics['PE_Ratio'])
                if metrics['PB_Ratio']:
                    sector_metrics['Avg_PB'].append(metrics['PB_Ratio'])
                if metrics['Dividend_Yield']:
                    sector_metrics['Avg_Dividend_Yield'].append(metrics['Dividend_Yield'])
                if metrics['Market_Cap']:
                    sector_metrics['Total_Market_Cap'].append(metrics['Market_Cap'])
                if metrics['Revenue_Growth']:
                    sector_metrics['Avg_Revenue_Growth'].append(metrics['Revenue_Growth'])
            
            # Calculate averages
            processed_fundamentals[sector] = {
                'Avg_PE': np.mean(sector_metrics['Avg_PE']) if sector_metrics['Avg_PE'] else None,
                'Avg_PB': np.mean(sector_metrics['Avg_PB']) if sector_metrics['Avg_PB'] else None,
                'Avg_Dividend_Yield': np.mean(sector_metrics['Avg_Dividend_Yield']) if sector_metrics['Avg_Dividend_Yield'] else None,
                'Total_Market_Cap': sum(sector_metrics['Total_Market_Cap']) if sector_metrics['Total_Market_Cap'] else None,
                'Avg_Revenue_Growth': np.mean(sector_metrics['Avg_Revenue_Growth']) if sector_metrics['Avg_Revenue_Growth'] else None
            }
            
        return processed_fundamentals
        
    def combine_all_data(self):
        """Combine all processed data into final dataset for model training"""
        if not self.load_data():
            return None
            
        # Process all data sources
        market_data = self.process_market_data()
        economic_data = self.process_economic_data()
        fundamental_data = self.process_fundamental_data()
        
        # Create combined dataset for each sector
        combined_data = {}
        
        for sector in market_data.keys():
            sector_df = market_data[sector].copy()
            
            # Add economic indicators
            for indicator, values in economic_data.items():
                if isinstance(values, pd.Series):
                    # Resample economic data to match market data frequency
                    resampled = values.resample('D').ffill()
                    # Align with market data dates
                    aligned = resampled.reindex(sector_df.index, method='ffill')
                    sector_df[indicator] = aligned
                    
            # Add fundamental data
            if sector in fundamental_data:
                for metric, value in fundamental_data[sector].items():
                    sector_df[metric] = value
                    
            combined_data[sector] = sector_df
            
        return combined_data
        
    def save_processed_data(self, data: Dict[str, pd.DataFrame], filename: str):
        """Save processed data to file"""
        try:
            # Convert DataFrames to dictionary format
            serializable_data = {k: v.to_dict(orient='split') for k, v in data.items()}
            
            with open(filename, 'w') as f:
                json.dump(serializable_data, f)
            print(f"Successfully saved processed data to {filename}")
            
        except Exception as e:
            print(f"Error saving processed data: {e}")
            
def main():
    processor = DataProcessor()
    combined_data = processor.combine_all_data()
    
    if combined_data:
        processor.save_processed_data(combined_data, 'enhanced_processed_data.json')
        print("Data processing completed successfully!")
    else:
        print("Error processing data!")
        
if __name__ == "__main__":
    main() 