# Aivestor AI Service

Flask-based machine learning service for stock predictions, portfolio analysis, and AI-powered chatbot.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # fill in your keys
python app.py                     # runs on port 5001
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single stock prediction |
| POST | `/predict/portfolio` | Portfolio-wide predictions |
| POST | `/chat` | AI chatbot (Gemini-powered) |
| POST | `/analyze` | Market analysis |

## Data Collection

Stock and market data is collected via `enhanced_data_collection.py`:

```bash
python enhanced_data_collection.py
```

This pulls data from Yahoo Finance, FRED, and news APIs. Output goes to `datacollection/` (gitignored — large files).

## Model Training

```bash
python train_enhanced_model_cv.py
```

Trains a cross-validated model using collected data. Saved models go to `models/`.

## Testing

```bash
pytest
```

## Project Structure

```
├── app.py                        # Flask API server
├── chatbot.py                    # Gemini chatbot logic
├── advanced_stock_predictor.py   # ML prediction engine
├── enhanced_data_collection.py   # Data pipeline
├── analyze_market_data.py        # Market analysis utilities
├── process_enhanced_data.py      # Data preprocessing
├── train_enhanced_model_cv.py    # Model training w/ cross-validation
├── models/                       # Trained model artifacts
├── tests/                        # pytest test suites
├── requirements.txt
└── .env.example
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Gemini API key |
| `FRED_API_KEY` | FRED economic data API key |
| `ALPHA_VANTAGE_KEY` | Alpha Vantage API key (optional) |
| `NEWS_API_KEY` | News API key (optional) |
| `PORT` | Server port (default 5001) |