# Advanced FAQ chatbot for Aivestor using TF-IDF and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List

class AivestorChatbot:
    # Initializing chatbot with FAQ data
    def __init__(self):
        self.faqs = [
            {
                'question': 'What is Aivestor?',
                'answer': 'Aivestor is an AI-powered stock prediction platform that provides short-term (63 days) and long-term (252 days) predictions for stocks and ETFs, along with portfolio recommendations.'
            },
            {
                'question': 'How accurate is Aivestor?',
                'answer': 'Aivestor achieves high accuracy, e.g., 94.12% for XRT long-term predictions, with cross-validation scores indicating robust performance.'
            },
            {
                'question': 'What data does Aivestor use?',
                'answer': 'Aivestor uses technical indicators (RSI, MACD, Bollinger Bands, ATR), VIX, sector sentiment, and economic data from FRED (e.g., GDP, unemployment).'
            },
            {
                'question': 'How do I get predictions for a stock?',
                'answer': 'Use the Flask API endpoint `/predict/<ticker>` (e.g., `/predict/AAPL`) or run `advanced_stock_predictor.py` with a ticker list.'
            },
            {
                'question': 'Can Aivestor predict sector trends?',
                'answer': 'Yes, Aivestor predicts sector trends (e.g., Technology via XLK) using the `/predict_sector/<sector>` endpoint or `predict_sector` method.'
            },
            {
                'question': 'How do I generate a portfolio?',
                'answer': 'Use the `/portfolio` endpoint with a POST request containing tickers and risk tolerance (low, medium, high), or call `generate_portfolio_recommendation`.'
            }
        ]
        self.vectorizer = TfidfVectorizer()
        self.faq_questions = [faq['question'].lower() for faq in self.faqs]
        self.question_vectors = self.vectorizer.fit_transform(self.faq_questions)

    # Finding the best matching FAQ for a user query
    def get_response(self, query: str) -> Dict:
        try:
            query = query.lower()
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
            best_idx = np.argmax(similarities)
            if similarities[best_idx] > 0.2:  # Threshold for relevance
                return {
                    'query': query,
                    'answer': self.faqs[best_idx]['answer'],
                    'confidence': float(similarities[best_idx])
                }
            else:
                return {
                    'query': query,
                    'answer': 'Sorry, I couldn’t find a relevant answer. Please try rephrasing your question.',
                    'confidence': 0.0
                }
        except Exception as e:
            return {
                'query': query,
                'answer': f'Error processing query: {e}',
                'confidence': 0.0
            }

if __name__ == "__main__":
    # Testing the chatbot
    chatbot = AivestorChatbot()
    queries = ['What is Aivestor?', 'How accurate is the model?', 'Unknown question']
    for query in queries:
        response = chatbot.get_response(query)
        print(f"Query: {response['query']}\nAnswer: {response['answer']}\nConfidence: {response['confidence']:.4f}\n")