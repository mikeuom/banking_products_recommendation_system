import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Initiate the Customer Segmenter class, utlizing the pre-trained Unsupervised K-Means model
# The same logic of the model is provided in the notebook file, refer to folder customer_segmentation_model in notebooks_models_development
class CustomerSegmenter:
    
    def __init__(self, model_path="models/customer_segment_model/kmeans_customer_segments.pkl"):
        self.model = joblib.load(model_path)
        
    def _clean_amount(self, amount):
        if amount is None or (isinstance(amount, float) and np.isnan(amount)):
            return 0.0
            
        try:
            if isinstance(amount, (int, float)):
                return float(amount)
                
            if isinstance(amount, str):
                cleaned_str = ''.join(c for c in amount if c.isdigit() or c == '.' or c == '-')
                if cleaned_str:
                    return float(cleaned_str)
                else:
                    return 0.0
                    
            return 0.0
        except:
            return 0.0
    
    def process_transactions(self, transactions_df):

        if transactions_df.empty:
            return pd.DataFrame(columns=['customer_id', 'recency', 'frequency', 'monetary'])
            
        df = transactions_df.copy()
        
        df['amount_cleaned'] = df['amount'].apply(lambda x: self._clean_amount(x) if pd.notna(x) else 0)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        df = df.dropna(subset=['date'])
        
        if df.empty or 'date' not in df.columns:
            return pd.DataFrame(columns=['customer_id', 'recency', 'frequency', 'monetary'])
        
        snapshot_date = df['date'].max()

        if pd.isna(snapshot_date):
            snapshot_date = pd.Timestamp.now()
        
        # RFM values
        rfm_df = df.groupby('customer_id').agg({
            'date': lambda x: (snapshot_date - x.max()).days,  
            'customer_id': 'count',                           
            'amount_cleaned': 'sum'                           
        }).rename(columns={
            'date': 'recency',
            'customer_id': 'frequency',
            'amount_cleaned': 'monetary'
        }).reset_index()
        
        rfm_df['recency'] = rfm_df['recency'].replace([np.inf, -np.inf], np.nan)
        rfm_df['recency'] = rfm_df['recency'].fillna(0)
        
        rfm_df['frequency'] = rfm_df['frequency'].replace([np.inf, -np.inf], np.nan)
        rfm_df['frequency'] = rfm_df['frequency'].fillna(0)
        
        rfm_df['monetary'] = rfm_df['monetary'].replace([np.inf, -np.inf], np.nan)
        rfm_df['monetary'] = rfm_df['monetary'].fillna(0)
        
        return rfm_df
    
    def predict_segment(self, rfm_df):
        features = rfm_df[['recency', 'frequency', 'monetary']].values
        
        clusters = self.model.predict(features)
        
        result_df = rfm_df.copy()
        result_df['cluster'] = clusters
        
        return result_df
    
    def get_recommendations(self, rfm_df):
        result_df = rfm_df.copy()
        
        def assign_rfm_scores(series, reverse=False):
            clean_series = series.copy()
            clean_series = clean_series.replace([np.inf, -np.inf], np.nan)
            if clean_series.isna().any():
                if reverse:
                    clean_series = clean_series.fillna(clean_series.min() if len(clean_series.dropna()) > 0 else 0)
                else:
                    clean_series = clean_series.fillna(clean_series.min() if len(clean_series.dropna()) > 0 else 0)
            
            unique_values = clean_series.nunique()
            
            if unique_values < 5:
                ranks = clean_series.rank(method='dense', ascending=not reverse)
                max_rank = ranks.max()
                if max_rank > 1:
                    scores = ((ranks - 1) / (max_rank - 1) * 4 + 1).round().astype(int)
                else:
                    scores = pd.Series(3, index=clean_series.index)
                return scores if not reverse else 6 - scores
            else:
                try:
                    labels = list(range(1, 6))
                    if reverse:
                        labels = list(reversed(labels))
                    return pd.qcut(clean_series, 5, labels=labels, duplicates='drop').astype(int)
                except:
                    ranks = clean_series.rank(method='dense', ascending=not reverse)
                    max_rank = ranks.max()
                    if max_rank > 1:
                        scores = ((ranks - 1) / (max_rank - 1) * 4 + 1).round().astype(int)
                    else:
                        scores = pd.Series(3, index=clean_series.index)
                    return scores if not reverse else 6 - scores
        
        # scoring
        result_df['R_score'] = assign_rfm_scores(result_df['recency'], reverse=True)  # Lower recency is better
        result_df['F_score'] = assign_rfm_scores(result_df['frequency'], reverse=False)  # Higher frequency is better
        result_df['M_score'] = assign_rfm_scores(result_df['monetary'], reverse=False)  # Higher monetary is better
        
        result_df['RFM_score'] = result_df[['R_score', 'F_score', 'M_score']].sum(axis=1)
        
        # Clusters 3 and 4: High-frequency, high-spending groups; strongly recommend credit cards
        # For clusters 1 and 2, frequency ≥ 16 or monetary ≥ 1000: Moderate to high spending power; recommend credit cards
        # For cluster 0, frequency ≥ 20: Low spending power; consider credit cards
        conditions = [
            (result_df['cluster'].isin([3, 4])),
            (result_df['cluster'].isin([1, 2]) & ((result_df['frequency'] >= 16) | (result_df['monetary'] >= 1000))),
            (result_df['cluster'] == 0) & (result_df['frequency'] >= 20),
            (True)
        ]
        choices = [
            'Strong Recommend',
            'Recommend',
            'Consider',
            'Not Recommend'
        ]
        result_df['credit_card_recommendation'] = np.select(conditions, choices, default='Not Recommend')
        
        # Direct Debit Recommendations based on RFM scores
        # Map RFM score to recommendation
        def map_direct_debit_recommend(score):
            if score >= 12:
                return 'Strong Recommend'
            elif score >= 9:
                return 'Recommend'
            elif score >= 6:
                return 'Consider'
            else:
                return 'Not Recommend'
        
        result_df['direct_debit_recommendation'] = result_df['RFM_score'].apply(map_direct_debit_recommend)
        
        return result_df
    
    def process_and_predict(self, transactions_df):
        rfm_df = self.process_transactions(transactions_df)
        segmented_df = self.predict_segment(rfm_df)
        result_df = self.get_recommendations(segmented_df)
        return result_df
    
    def predict_for_new_customer(self, recency, frequency, monetary):
        features = np.array([[recency, frequency, monetary]])
        
        cluster = self.model.predict(features)[0]
        
        r_score = 5 if recency < 30 else (4 if recency < 60 else (3 if recency < 90 else (2 if recency < 120 else 1)))
        f_score = 1 if frequency < 5 else (2 if frequency < 10 else (3 if frequency < 15 else (4 if frequency < 20 else 5)))
        m_score = 1 if monetary < 200 else (2 if monetary < 500 else (3 if monetary < 800 else (4 if monetary < 1200 else 5)))
        
        rfm_score = r_score + f_score + m_score
        
        # recommendations
        credit_card_recommendation = 'Strong Recommend' if cluster in [3, 4] else \
                                    ('Recommend' if (cluster in [1, 2] and (frequency >= 16 or monetary >= 1000)) else \
                                    ('Consider' if (cluster == 0 and frequency >= 20) else 'Not Recommend'))
        
        direct_debit_recommendation = 'Strong Recommend' if rfm_score >= 12 else \
                                     ('Recommend' if rfm_score >= 9 else \
                                     ('Consider' if rfm_score >= 6 else 'Not Recommend'))
        
        return {
            'cluster': int(cluster),
            'RFM_score': int(rfm_score),
            'credit_card_recommendation': credit_card_recommendation,
            'direct_debit_recommendation': direct_debit_recommendation
        } 