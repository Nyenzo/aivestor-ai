# Aivestor AI 🤖📈

An advanced AI-powered investment advisory system that combines machine learning, sentiment analysis, and economic indicators to provide intelligent stock market predictions and portfolio recommendations.

## 🌟 Features

- **Advanced Stock Prediction**: Utilizes machine learning models to forecast stock movements
- **Sentiment Analysis**: Incorporates market sentiment from news and social media
- **Economic Indicator Analysis**: Integrates FRED economic data for comprehensive market analysis
- **Portfolio Optimization**: Provides personalized portfolio recommendations based on risk tolerance
- **Real-time Data Processing**: Continuous updates from multiple financial data sources
- **RESTful API**: Easy integration with frontend applications

## 🛠️ Technology Stack

- **AI/ML**: Python, scikit-learn, PyTorch, Transformers
- **Data Processing**: NumPy, Pandas, yfinance
- **API Integration**: Alpha Vantage, FRED, News API
- **Backend**: Flask REST API
- **Visualization**: Matplotlib, Seaborn
- **Text Processing**: TextBlob, PyPDF2

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL
- API Keys:
  - Alpha Vantage
  - FRED
  - News API

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/aivestor-ai.git
   cd aivestor-ai
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   FRED_API_KEY=your_key_here
   NEWS_API_KEY=your_key_here
   ```

5. **Initialize database**
   ```bash
   psql -U your_username -d your_database -f setup_tables.sql
   ```

6. **Start the Flask server**
   ```bash
   python app.py
   ```

## 🔄 API Endpoints

### Health Check
```
GET /
```

### Predict Stock Movement
```
POST /api/predict
{
    "sector": "technology",
    "ticker": "AAPL",
    "sentiment_score": 0.75
}
```

### Portfolio Recommendations
```
POST /api/portfolio/recommend
{
    "predictions": {...},
    "risk_tolerance": "moderate"
}
```

### Train Model
```
POST /api/model/train
{
    "sector_data": {...},
    "sentiment_results": {...}
}
```

## 📊 Data Collection and Processing

The system collects and processes data through multiple components:
- `enhanced_data_collection.py`: Gathers financial and market data
- `process_enhanced_data.py`: Preprocesses and transforms raw data
- `train_enhanced_model_cv.py`: Trains models with cross-validation
- `advanced_stock_predictor.py`: Core prediction engine

## 📈 Visualization Examples

The system generates various analytical visualizations:
- Correlation matrices
- Sentiment analysis trends
- Economic indicators
- Volatility comparisons
- Sector performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.