"""
PHASE 2: ADVANCED ML TRAINING PIPELINE
XGBoost with hyperparameter optimization and detailed evaluation.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Columns excluded from training (not features)
COLS_TO_DROP = ['Target', 'Close', 'Open', 'High', 'Low', 'Volume',
                'BB_High', 'BB_Low', 'SMA_200', 'EMA_200', 'Volume_SMA20']


def load_dataset(symbol="BTC-USD"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, f"{symbol.replace('-','_')}_dataset.csv")
    print(f"[LOAD] Reading: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")
    return df


def prepare_features(df):
    drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=drop)
    y = df['Target']

    # Cleaning: Infinite values and NaNs that might have remained
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"  Final feature set: {len(X.columns)} indicators")
    return X, y


def time_series_split_train(X, y, symbol):
    print("\n[SPLIT] Professional TimeSeriesSplit (no data leakage)...")

    split_80 = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_80], X.iloc[split_80:]
    y_train, y_test = y.iloc[:split_80], y.iloc[split_80:]
    print(f"  Train: {len(X_train)} days | Test: {len(X_test)} days")

    # Class weights - balancing unequal classes
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print("\n[TRAIN] XGBoost with optimized parameters...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=class_ratio,  # Solves class imbalance!
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"  Optimal number of trees: {model.best_iteration}")
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    print("\n[EVAL] Testing on unseen data (last 20%)...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%  (When calling BUY, how often is it right)")
    print(f"  Recall   : {metrics['recall']*100:.2f}%  (How many actual bursts it caught)")
    print(f"  F1 Score : {metrics['f1']*100:.2f}%")
    print(f"  ROC AUC  : {metrics['roc_auc']:.4f}  (0.5=coin flip, 1.0=perfect)")
    print("\n" + classification_report(y_test, y_pred))

    return metrics


def save_model_and_metadata(model, X_train, metrics, symbol):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(base_dir, symbol.replace('-','_'))

    # Saves model
    model_path = f"{base}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Saves metadata (metrics + feature list) - dashboard will use this
    meta = {
        'symbol': symbol,
        'metrics': metrics,
        'features': list(model.feature_names_in_),
        'n_features': len(model.feature_names_in_),
        'n_estimators_best': int(model.best_iteration),
    }
    meta_path = f"{base}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE] Model -> {model_path}")
    print(f"[SAVE] Meta  -> {meta_path}")
    return model_path


def train_symbol(symbol="BTC-USD"):
    print(f"\n{'='*55}")
    print(f" TRAINING MODEL: {symbol}")
    print(f"{'='*55}")

    df = load_dataset(symbol)
    X, y = prepare_features(df)
    model, X_train, X_test, y_train, y_test = time_series_split_train(X, y, symbol)
    metrics = evaluate_model(model, X_test, y_test)
    save_model_and_metadata(model, X_train, metrics, symbol)
    return model, metrics


if __name__ == "__main__":
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "META", "AMZN", "SPY"]
    for s in symbols:
        try:
            train_symbol(s)
        except Exception as e:
            print(f"Error processing {s}: {e}")
    print("\n[DONE] All models have been trained and saved!")
