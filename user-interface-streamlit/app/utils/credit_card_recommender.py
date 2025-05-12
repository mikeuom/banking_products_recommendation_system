import pandas as pd
import joblib
import os
import numpy as np

# Initiate the Credit Card Recommender class, utlizing the pre-trained Unsupervised K-Means model
# The same logic of the model is provided in the notebook file, refer to folder credit_card_products_prediction_model in notebooks_models_development

class CreditCardRecommender():
    def __init__(self, model_path):
        self.model = None
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")

        self.credit_card_categories = {
            "FUEL":       [5541,4784,7538,7542,7531,7549],
            "RETAIL":     [5311,5310,5300,5651,5621,5661,5977,5912,5942,5941,5947,
                        5732,5733,5722,7230,5655,5921,5970,5712,5932,5193,3504,
                        5261,3132],
            "TRAVEL":     [4511,4411,4722,7011,4111,4112,4131,3722,3771,4121],
            "ENTERTAINMENT":[5812,5814,7996,7832,5813,7922,7801,7802,5816,5815,
                            4899,7995,5192],
            "GROCERIES":  [5411,5499],
            "OTHER":      [41800,4829,5211,3780,5719,4814,8099,8021,33180,5251,
                        3596,3730,9402,6300,7349,3775,3684,5045,8041,8011,
                        4214,3509,7210,3640,7393,8111,5094,3389,8043,3393,
                        3174,3001,3395,3058,8049,3387,3405,3144,3359,8931,
                        8062,7276,3260,3256,3006,1711,3007,3075,3066,3005,
                        3000,5533,3008,3009]
        }

        self.expected_categories = ['FUEL', 'RETAIL', 'TRAVEL', 'ENTERTAINMENT', 'GROCERIES', 'OTHER', 'UNMAPPED']

        self.card_map = {
            'FUEL':         'Fuel Rewards Card',
            'TRAVEL':       'Travel Rewards Card',
            'GROCERIES':    'Grocery Cashback Card',
            'RETAIL':       'Retail Cashback Card',
            'ENTERTAINMENT':'Dining & Entertainment Card',
            'OTHER':        'General Purpose Card',
            'UNMAPPED':     'Standard Card'
        }
    
    def aggregate_spend_wide(self, transactions_df):
        mcc_to_cat = {mcc: cat for cat, codes in self.credit_card_categories.items() for mcc in codes}
        transactions_df['spend_cat'] = transactions_df['mcc'].map(mcc_to_cat).fillna('UNMAPPED')
        
        agg = (
            transactions_df.groupby(['customer_id', 'spend_cat'])
            .agg(total_amt=('amount', 'sum'), txt_count=('amount', 'count'))
            .reset_index()
        )

        spend_wide = (
            agg
            .pivot_table(index='customer_id', 
                         columns='spend_cat', 
                         values='total_amt', 
                         fill_value=0)
            .add_prefix('spend_')
            .reset_index()
        )

        spend_cols = [c for c in spend_wide.columns if c.startswith('spend_')]
        spend_wide['total_spend'] = spend_wide[spend_cols].sum(axis=1)
        for c in spend_cols:
            spend_wide[c + '_pct'] = spend_wide[c] / spend_wide['total_spend']

        return spend_wide
    
    def feature_selection_scaling(self, spend_wide):
        # fill with 0 if missing categories
        for category in self.expected_categories:
            pct_col = f'spend_{category}_pct'
            if pct_col not in spend_wide.columns:
                spend_wide[pct_col] = 0
                
        features = [f'spend_{cat}_pct' for cat in self.expected_categories]
        X = spend_wide[features].fillna(0)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        scaler = joblib.load(os.path.join(dir_path, "../../models/credit_card_model/creditcard_scaler.joblib"))
        X_array = X.values
        X_scaled = scaler.transform(X_array)

        return X_scaled
    
    def predict_credit_card(self, spend_wide):
        X_scaled = self.feature_selection_scaling(spend_wide)
        cluster = self.model.predict(X_scaled)
        spend_wide['cluster'] = cluster

        features = [f'spend_{cat}_pct' for cat in self.expected_categories]
        segment_profile = spend_wide.copy()
        segment_profile['dominant_col'] = segment_profile[features].idxmax(axis=1)
        segment_profile['dominant_cat'] = (
            segment_profile['dominant_col']
            .str.replace('spend_','')
            .str.replace('_pct','')
        )


        segment_profile['recommended_card'] = segment_profile['dominant_cat'].map(self.card_map)
        
        # Create cluster-to-category mapping
        cluster_to_category = {}
        for cluster_id in segment_profile['cluster'].unique():
            cluster_rows = segment_profile[segment_profile['cluster'] == cluster_id]
            most_common_category = cluster_rows['dominant_cat'].mode()[0]
            cluster_to_category[cluster_id] = most_common_category
        
        cluster_to_card = {cluster_id: self.card_map[category] 
                         for cluster_id, category in cluster_to_category.items()}
        
        spend_wide['recommended_card'] = spend_wide['cluster'].map(cluster_to_card)
        
        # affinity score based on cluster's dominant category
        spend_wide['recommendation_score'] = spend_wide.apply(
            lambda r: r[f"spend_{cluster_to_category[r['cluster']]}_pct"], 
            axis=1
        )

        threshold = 0.30
        spend_wide['final_offer'] = spend_wide.apply(
            lambda r: r['recommended_card'] if r['recommendation_score'] >= threshold
                    else 'No Targeted Offer',
            axis=1
        )

        spend_wide = spend_wide[['customer_id','cluster','recommendation_score','final_offer']]
        
        return spend_wide
    




def main():

     # Get the input for testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transactions_path = os.path.join(script_dir, "../../data/demo_data/transactions_demo.csv")
    transactions = pd.read_csv(transactions_path)
    customer_id = 1405924
    transactions_df = transactions[transactions['customer_id'] == customer_id]
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, "../../models/credit_card_model/kmeans_model_creditcardsubtypeunsupervised.pkl")

    credit_card_recommender = CreditCardRecommender(model_path)
    spend_wide = credit_card_recommender.aggregate_spend_wide(transactions_df)
    result = credit_card_recommender.predict_credit_card(spend_wide)

    print(result)
    
if __name__ == "__main__":
    main()