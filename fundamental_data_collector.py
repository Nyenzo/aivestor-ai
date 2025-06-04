import os
import json
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv
import time
import pandas as pd
from fredapi import Fred

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not ALPHA_VANTAGE_KEY:
    print("Warning: ALPHA_VANTAGE_KEY not found in .env file. Will use yfinance data only.")
if not FRED_API_KEY:
    print("Warning: FRED_API_KEY not found in .env file. Will skip economic indicators.")

class FundamentalDataCollector:
    def __init__(self):
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Utilities': 'XLU'
        }
        
        self.sector_tickers = {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "ADBE"],
            "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "MPC"],
            "Industrials": ["CAT", "BA", "HON", "UNP", "MMM", "GE"],
            "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP"]
        }
        
        # Enhanced FRED indicators
        self.fred_indicators = {
            'GDP': 'GDP',                           # Gross Domestic Product
            'Real_GDP': 'GDPC1',                    # Real GDP
            'Inflation': 'CPIAUCSL',                # Consumer Price Index
            'Core_Inflation': 'CPILFESL',           # Core CPI (excluding food and energy)
            'Unemployment': 'UNRATE',               # Unemployment Rate
            'Initial_Claims': 'ICSA',               # Initial Jobless Claims
            'Nonfarm_Payrolls': 'PAYEMS',          # Total Nonfarm Payrolls
            'Fed_Funds_Rate': 'FEDFUNDS',          # Federal Funds Rate
            '10Y_Treasury': 'DGS10',               # 10-Year Treasury Rate
            '2Y_Treasury': 'DGS2',                 # 2-Year Treasury Rate
            'Industrial_Production': 'INDPRO',      # Industrial Production
            'Consumer_Sentiment': 'UMCSENT',        # Consumer Sentiment
            'Retail_Sales': 'RSAFS',               # Retail Sales
            'Housing_Starts': 'HOUST',             # Housing Starts
            'PCE': 'PCE',                          # Personal Consumption Expenditures
            'Capacity_Utilization': 'TCU',         # Capacity Utilization
            'Labor_Force_Participation': 'CIVPART'  # Labor Force Participation Rate
        }
        
        if FRED_API_KEY:
            self.fred = Fred(api_key=FRED_API_KEY)

    def collect_yfinance_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Collect fundamental data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cash_flow
            
            fundamentals = {
                # Valuation Metrics
                'PE_Ratio': info.get('forwardPE'),
                'PB_Ratio': info.get('priceToBook'),
                'PS_Ratio': info.get('priceToSalesTrailing12Months'),
                'PEG_Ratio': info.get('pegRatio'),
                
                # Financial Health
                'Debt_to_Equity': info.get('debtToEquity'),
                'Current_Ratio': info.get('currentRatio'),
                'Quick_Ratio': info.get('quickRatio'),
                
                # Profitability
                'ROE': info.get('returnOnEquity'),
                'ROA': info.get('returnOnAssets'),
                'Profit_Margin': info.get('profitMargins'),
                'Operating_Margin': info.get('operatingMargins'),
                
                # Growth & Performance
                'EPS': info.get('trailingEPS'),
                'EPS_Growth': info.get('earningsGrowth'),
                'Revenue_Growth': info.get('revenueGrowth'),
                
                # Dividend Information
                'Dividend_Yield': info.get('dividendYield'),
                'Dividend_Rate': info.get('dividendRate'),
                'Payout_Ratio': info.get('payoutRatio'),
                
                # Cash Flow Analysis
                'Free_Cash_Flow': cash_flow.loc['Free Cash Flow'].iloc[0] if not cash_flow.empty else None,
                'Operating_Cash_Flow': cash_flow.loc['Operating Cash Flow'].iloc[0] if not cash_flow.empty else None,
                
                # Market Position
                'Market_Cap': info.get('marketCap'),
                'Enterprise_Value': info.get('enterpriseValue'),
                'Beta': info.get('beta'),
                
                # Additional Metrics
                'Shares_Outstanding': info.get('sharesOutstanding'),
                'Float_Shares': info.get('floatShares'),
                'Short_Ratio': info.get('shortRatio'),
                
                # Company Info
                'Industry': info.get('industry'),
                'Sector': info.get('sector'),
                'Business_Summary': info.get('longBusinessSummary'),
                'Full_Time_Employees': info.get('fullTimeEmployees'),
                'Website': info.get('website')
            }
            
            return fundamentals
            
        except Exception as e:
            print(f"Error collecting yfinance fundamentals for {ticker}: {e}")
            return None

    def collect_alpha_vantage_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Collect fundamental data from Alpha Vantage"""
        if not ALPHA_VANTAGE_KEY:
            print("Alpha Vantage API key not found")
            return None
            
        try:
            # Overview
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}'
            r = requests.get(url)
            overview = r.json()
            
            # Income Statement
            url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}'
            r = requests.get(url)
            income = r.json()
            
            # Balance Sheet
            url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}'
            r = requests.get(url)
            balance = r.json()
            
            # Cash Flow
            url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}'
            r = requests.get(url)
            cash_flow = r.json()
            
            fundamentals = {
                # From Overview
                'Market_Cap': float(overview.get('MarketCapitalization', 0)),
                'PE_Ratio': float(overview.get('PERatio', 0)),
                'PEG_Ratio': float(overview.get('PEGRatio', 0)),
                'Dividend_Yield': float(overview.get('DividendYield', 0)),
                'Profit_Margin': float(overview.get('ProfitMargin', 0)),
                'ROE': float(overview.get('ReturnOnEquityTTM', 0)),
                'ROA': float(overview.get('ReturnOnAssetsTTM', 0)),
                'Beta': float(overview.get('Beta', 0)),
                
                # From Income Statement (most recent annual)
                'Revenue': float(income.get('annualReports', [{}])[0].get('totalRevenue', 0)),
                'Gross_Profit': float(income.get('annualReports', [{}])[0].get('grossProfit', 0)),
                'Operating_Income': float(income.get('annualReports', [{}])[0].get('operatingIncome', 0)),
                'Net_Income': float(income.get('annualReports', [{}])[0].get('netIncome', 0)),
                
                # From Balance Sheet (most recent annual)
                'Total_Assets': float(balance.get('annualReports', [{}])[0].get('totalAssets', 0)),
                'Total_Liabilities': float(balance.get('annualReports', [{}])[0].get('totalLiabilities', 0)),
                'Total_Equity': float(balance.get('annualReports', [{}])[0].get('totalShareholderEquity', 0)),
                
                # From Cash Flow (most recent annual)
                'Operating_Cash_Flow': float(cash_flow.get('annualReports', [{}])[0].get('operatingCashflow', 0)),
                'Capital_Expenditure': float(cash_flow.get('annualReports', [{}])[0].get('capitalExpenditures', 0))
            }
            
            # Calculate additional ratios
            if fundamentals['Total_Assets'] and fundamentals['Total_Liabilities']:
                fundamentals['Debt_to_Assets'] = fundamentals['Total_Liabilities'] / fundamentals['Total_Assets']
            
            if fundamentals['Total_Equity'] and fundamentals['Total_Liabilities']:
                fundamentals['Debt_to_Equity'] = fundamentals['Total_Liabilities'] / fundamentals['Total_Equity']
            
            if fundamentals['Operating_Cash_Flow'] and fundamentals['Capital_Expenditure']:
                fundamentals['Free_Cash_Flow'] = fundamentals['Operating_Cash_Flow'] - abs(fundamentals['Capital_Expenditure'])
            
            return fundamentals
            
        except Exception as e:
            print(f"Error collecting Alpha Vantage fundamentals for {ticker}: {e}")
            return None

    def collect_fred_data(self) -> Dict[str, pd.Series]:
        """Collect economic indicators from FRED"""
        if not FRED_API_KEY:
            print("FRED API key not found. Skipping economic indicators.")
            return {}
            
        economic_data = {}
        start_date = datetime.now() - timedelta(days=365*5)  # 5 years of data
        
        for indicator_name, series_id in self.fred_indicators.items():
            try:
                print(f"Collecting {indicator_name} from FRED...")
                series = self.fred.get_series(series_id, observation_start=start_date)
                economic_data[indicator_name] = series
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error collecting {indicator_name}: {e}")
        
        # Calculate additional derived indicators
        if 'Fed_Funds_Rate' in economic_data and '10Y_Treasury' in economic_data:
            economic_data['Yield_Curve_Spread'] = economic_data['10Y_Treasury'] - economic_data['Fed_Funds_Rate']
        
        if 'GDP' in economic_data:
            economic_data['GDP_Growth'] = economic_data['GDP'].pct_change()
        
        if 'Nonfarm_Payrolls' in economic_data:
            economic_data['Employment_Change'] = economic_data['Nonfarm_Payrolls'].pct_change()
        
        # Save economic data to separate file
        econ_file = f'economic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            # Convert to dictionary of lists for JSON serialization
            econ_dict = {k: v.to_dict() for k, v in economic_data.items()}
            with open(econ_file, 'w') as f:
                json.dump(econ_dict, f, indent=4)
            print(f"Economic data saved to {econ_file}")
        except Exception as e:
            print(f"Error saving economic data: {e}")
        
        return economic_data

    def collect_all_fundamentals(self) -> Dict[str, Dict[str, Any]]:
        """Collect fundamental data for all tickers"""
        # First collect economic data
        print("\nCollecting economic indicators from FRED...")
        economic_data = self.collect_fred_data()
        
        fundamentals = {}
        
        for sector, tickers in self.sector_tickers.items():
            print(f"\nCollecting fundamental data for {sector} sector...")
            sector_fundamentals = {}
            
            for ticker in tickers:
                print(f"Processing {ticker}...")
                
                # Always collect yfinance data
                yf_data = self.collect_yfinance_fundamentals(ticker)
                time.sleep(1)  # Avoid rate limiting
                
                # Only try Alpha Vantage if key is available
                if ALPHA_VANTAGE_KEY:
                    av_data = self.collect_alpha_vantage_fundamentals(ticker)
                    time.sleep(12)  # Alpha Vantage rate limit is 5 calls per minute
                else:
                    av_data = None
                    print(f"Skipping Alpha Vantage data collection for {ticker} (no API key)")
                
                # Combine data, preferring Alpha Vantage when available
                combined_data = {}
                if yf_data:
                    combined_data.update(yf_data)
                if av_data:
                    combined_data.update(av_data)
                
                # Add relevant economic indicators
                if economic_data:
                    latest_econ = {k: v.iloc[-1] for k, v in economic_data.items()}
                    combined_data['Economic_Context'] = latest_econ
                
                if combined_data:
                    sector_fundamentals[ticker] = combined_data
                    print(f"Successfully collected fundamental data for {ticker}")
                    if not av_data:
                        print("Note: Using yfinance data only")
                else:
                    print(f"No fundamental data collected for {ticker}")
            
            fundamentals[sector] = sector_fundamentals
        
        # Save to JSON file
        output_file = f'fundamental_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(fundamentals, f, indent=4)
            print(f"\nFundamental data saved to {output_file}")
        except Exception as e:
            print(f"Error saving fundamental data: {e}")
        
        return fundamentals

def main():
    collector = FundamentalDataCollector()
    fundamentals = collector.collect_all_fundamentals()
    
    # Print summary
    print("\nFundamental Data Collection Summary:")
    for sector, data in fundamentals.items():
        print(f"\n{sector}:")
        for ticker, fundamentals in data.items():
            metrics = [k for k, v in fundamentals.items() if v is not None]
            print(f"{ticker}: {len(metrics)} metrics collected")
            if 'Economic_Context' in fundamentals:
                print(f"  Including {len(fundamentals['Economic_Context'])} economic indicators")

if __name__ == "__main__":
    main() 