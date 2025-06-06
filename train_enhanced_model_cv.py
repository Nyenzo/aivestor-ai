# Training and evaluating machine learning models with cross-validation
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class EnhancedModelTrainer:
    DATA_DIR = 'datacollection'
    MODEL_DIR = 'models'

    # Initializing the trainer with processed features path
    def __init__(self, processed_features_path: str):
        self.processed_features_path = processed_features_path
        self.features = self.load_features()
        self.short_term_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.long_term_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    # Loading processed features from pickle file
    def load_features(self) -> Dict[str, pd.DataFrame]:
        try:
            with open(self.processed_features_path, 'rb') as f:
                features = pickle.load(f)
            print(f"Loaded processed features: {len(features)} tickers")
            return features
        except Exception as e:
            print(f"Error loading features: {e}")
            return {}

    # Preparing features and target for a ticker
    def prepare_data(self, df: pd.DataFrame, target: str, is_etf: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        try:
            base_features = ['Close', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'VIX', 'Sector_Sentiment']
            economic_features = [
                'GDP', 'Real_GDP', 'Inflation', 'Core_Inflation', 'Unemployment', 'Initial_Claims',
                'Nonfarm_Payrolls', 'Fed_Funds_Rate', '10Y_Treasury', '2Y_Treasury', 'Industrial_Production',
                'Consumer_Sentiment', 'Retail_Sales', 'Housing_Starts', 'PCE', 'Capacity_Utilization',
                'Labor_Force_Participation', 'Yield_Curve_Spread', 'GDP_Growth', 'Employment_Change'
            ]
            fundamental_features = ['PE_Ratio', 'EPS', 'Revenue_TTM', 'Debt_to_Equity']
            
            feature_names = base_features + economic_features
            if not is_etf:
                feature_names += fundamental_features

            if target not in df.columns:
                print(f"Target column {target} not found in DataFrame")
                return np.array([]), np.array([]), []

            # Check class distribution
            y = df[target].map({'Buy': 1, 'Sell': 0, 'Hold': 2})
            class_counts = pd.Series(y).value_counts()
            if any(class_counts < 3):
                print(f"Skipping {target}: insufficient samples {class_counts.to_dict()}")
                return np.array([]), np.array([]), []

            X = df[feature_names].fillna(0.0)
            print(f"Prepared data: {X.shape[0]} rows, {len(feature_names)} features")
            return X.values, y.values, feature_names
        except Exception as e:
            print(f"Error preparing data: {e}")
            return np.array([]), np.array([]), []

    # Printing performance metrics
    def print_metrics(self, ticker: str, model_type: str, results: Dict):
        if not results:
            return
        print(f"\n{ticker} {model_type} Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Cross-Validation Scores: Mean = {results['cv_scores'].mean():.4f}, Std = {results['cv_scores'].std():.4f}")
        print("Classification Report:")
        for label, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"  {label}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1-Score = {metrics['f1-score']:.4f}")

    # Training and evaluating models for a ticker
    def train_for_ticker(self, ticker: str, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        try:
            print(f"\nTraining for {ticker}")
            is_etf = ticker in ['XLK', 'XLE', 'XLY', 'XLP', 'XLF', 'XLU', 'XLRE', 'XLI', 'XLB', 'XLV', 'XLC', 'XRT', 'XAUUSD']
            
            # Short-term model
            X_short, y_short, feature_names = self.prepare_data(df, 'Short_Term_Signal', is_etf)
            if X_short.size == 0 or y_short.size == 0:
                print(f"No valid data for {ticker} short-term model")
                return {}, {}
            
            X_train, X_test, y_train, y_test = train_test_split(X_short, y_short, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.short_term_model.fit(X_train_scaled, y_train)
            short_pred = self.short_term_model.predict(X_test_scaled)
            short_scores = cross_val_score(self.short_term_model, X_short, y_short, cv=3)
            
            short_results = {
                'accuracy': accuracy_score(y_test, short_pred),
                'cv_scores': short_scores,
                'classification_report': classification_report(y_test, short_pred, output_dict=True, zero_division=0),
                'feature_importance': pd.Series(self.short_term_model.feature_importances_, index=feature_names).to_dict()
            }
            self.print_metrics(ticker, "Short-Term", short_results)
            
            # Long-term model
            X_long, y_long, _ = self.prepare_data(df, 'Long_Term_Signal', is_etf)
            if X_long.size == 0 or y_long.size == 0:
                print(f"No valid data for {ticker} long-term model")
                return short_results, {}
            
            X_train, X_test, y_train, y_test = train_test_split(X_long, y_long, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.long_term_model.fit(X_train_scaled, y_train)
            long_pred = self.long_term_model.predict(X_test_scaled)
            long_scores = cross_val_score(self.long_term_model, X_long, y_long, cv=3)
            
            long_results = {
                'accuracy': accuracy_score(y_test, long_pred),
                'cv_scores': long_scores,
                'classification_report': classification_report(y_test, long_pred, output_dict=True, zero_division=0)
            }
            self.print_metrics(ticker, "Long-Term", long_results)
            
            return short_results, long_results
        except Exception as e:
            print(f"Error training for {ticker}: {e}")
            return {}, {}

    # Plotting ROC curves
    def plot_roc(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str):
        try:
            fpr, tpr, _ = roc_curve(y_true == 1, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Error plotting ROC: {e}")

    # Saving trained models and scaler
    def save_models(self):
        try:
            with open(os.path.join(self.MODEL_DIR, 'short_term_model.pkl'), 'wb') as f:
                pickle.dump(self.short_term_model, f)
            with open(os.path.join(self.MODEL_DIR, 'long_term_model.pkl'), 'wb') as f:
                pickle.dump(self.long_term_model, f)
            with open(os.path.join(self.MODEL_DIR, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            print("Saved models and scaler")
        except Exception as e:
            print(f"Error saving models: {e}")

    # Training and evaluating all tickers
    def train_and_evaluate(self):
        print("Starting training and evaluation")
        results = {}
        plt.figure(figsize=(10, 6))
        
        if not self.features:
            print("No features loaded, aborting training")
            return
        
        for ticker, df in self.features.items():
            if df.empty:
                print(f"Skipping {ticker}: empty DataFrame")
                continue
            short_results, long_results = self.train_for_ticker(ticker, df)
            if short_results or long_results:
                results[ticker] = {'short_term': short_results, 'long_term': long_results}
                
                if short_results:
                    X_short, y_short, _ = self.prepare_data(df, 'Short_Term_Signal', ticker in ['XLK', 'XLE', 'XLY', 'XLP', 'XLF', 'XLU', 'XLRE', 'XLI', 'XLB', 'XLV', 'XLC', 'XRT', 'XAUUSD'])
                    if X_short.size > 0:
                        X_scaled = self.scaler.transform(X_short)
                        y_pred_proba = self.short_term_model.predict_proba(X_scaled)
                        self.plot_roc(y_short, y_pred_proba, f'{ticker}_short')
        
        if results:
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.savefig(os.path.join(self.MODEL_DIR, 'roc_curves.png'))
            plt.close()
            
            self.save_models()
            
            # Aggregating feature importance for short-term model
            feat_importance = None
            for ticker, res in results.items():
                if 'feature_importance' in res['short_term']:
                    if feat_importance is None:
                        feat_importance = pd.Series(res['short_term']['feature_importance'])
                    else:
                        feat_importance += pd.Series(res['short_term']['feature_importance'])
            if feat_importance is not None:
                feat_importance /= len(results)
                plt.figure(figsize=(12, 6))
                feat_importance.sort_values(ascending=False).plot(kind='bar')
                plt.title('Feature Importance (Short-Term Model)')
                plt.savefig(os.path.join(self.MODEL_DIR, 'feature_importance.png'))
                plt.close()
            
            # Summarizing overall performance
            short_acc = [res['short_term']['accuracy'] for res in results.values() if res['short_term']]
            long_acc = [res['long_term']['accuracy'] for res in results.values() if res['long_term']]
            short_cv = [res['short_term']['cv_scores'].mean() for res in results.values() if res['short_term']]
            long_cv = [res['long_term']['cv_scores'].mean() for res in results.values() if res['long_term']]
            
            print("\nOverall Performance Summary:")
            if short_acc:
                print(f"Short-Term Model: Avg Accuracy = {np.mean(short_acc):.4f}, Avg CV Score = {np.mean(short_cv):.4f}")
            if long_acc:
                print(f"Long-Term Model: Avg Accuracy = {np.mean(long_acc):.4f}, Avg CV Score = {np.mean(long_cv):.4f}")
            print(f"Training complete. Results saved for {len(results)} tickers")
        else:
            print("No models trained due to errors")

if __name__ == "__main__":
    # Executing training and evaluation
    trainer = EnhancedModelTrainer(os.path.join('datacollection', 'processed_features.pkl'))
    trainer.train_and_evaluate()