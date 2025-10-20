import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path
import pickle

np.random.seed(42)
num_samples = 500
features = {
    'transaction_amount': np.random.uniform(10, 2000, num_samples),
    'transaction_time_seconds': np.random.randint(0, 30 * 24 * 3600, num_samples),
    'is_weekend': np.random.randint(0, 2, num_samples),
    'hour_of_day': np.random.randint(0, 24, num_samples),
    'is_round_amount': np.random.randint(0, 2, num_samples),
    'unique_receivers_24h': np.random.randint(0, 50, num_samples),
    'vpn_detected': np.random.randint(0, 2, num_samples),
    'location_risk_score': np.random.uniform(0, 1, num_samples),
    'transaction_frequency_30min': np.random.randint(0, 10, num_samples),
    'login_ip_changed_last_hour': np.random.randint(0, 2, num_samples),
    'avg_transaction_amount_24h': np.random.uniform(20, 1500, num_samples),
    'time_since_last_tx': np.random.randint(0, 86400, num_samples),
    'ip_risk_score': np.random.uniform(0, 1, num_samples),
    'transactions_last_24h': np.random.randint(0, 30, num_samples),
    'customer_segment': np.random.choice(['standard', 'premium', 'business'], size=num_samples, p=[0.7, 0.2, 0.1]),
}

df = pd.DataFrame(features)
y = ((df['transaction_amount'] > 1500) & (df['hour_of_day'] < 6) & (df['location_risk_score'] > 0.7)).astype(int)
X = df

dense_features = [c for c in X.columns if c != 'customer_segment']
cat_features = ['customer_segment']
preprocess = ColumnTransformer([
    ('num', StandardScaler(), dense_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('clf', LogisticRegression(max_iter=500))
])
pipeline.fit(X, y)
probs = pipeline.predict_proba(X)[:, 1]
auc = roc_auc_score(y, probs)
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
with open(models_dir / 'fraud_detection_advanced_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
info = {
    'model_version': '2.0.0',
    'algorithm': 'LogisticRegression',
    'feature_names': list(X.columns),
    'metrics': {
        'auc': auc,
        'samples': int(num_samples)
    }
}
with open(models_dir / 'model_performance.json', 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2)
