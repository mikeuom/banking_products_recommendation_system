import os
import joblib
import pandas as pd
import numpy as np
# from feature_engineering import FeatureEngineer

# Initiate the Banking Product Category Recommender class, utlizing the pre-trained Multi-Label Classification model (XGBoost)
# The same logic of the model is provided in the notebook file, refer to folder banking_products_category_recommendation_model in notebooks_models_development

class ProductCategoryRecommender():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        try:
            self.model = self.load_model()
        except Exception as e:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.labels_col = [
            "saving_account", "guanrantees", "junior_account", "loans", "pension"
        ]

        self.custom_threshhold = {
            'saving_account': 0.3,
            'guarantees': 0.5,
            'junior_account': 0.5,
            'loans': 0.5,
            'pension': 0.3,
        }


    def load_model(self):
        model_path = self.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        else:
            return joblib.load(model_path)
        

    def preprocess_data(self, customer_df):
        df = customer_df.copy()

        # Convert categorical columns to category type
        cat_cols = ['residence_country', 'residence_index', 'channel_entrace']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

        # Derive membership_days from first_join_date
        if 'first_join_date' in df.columns:
            join_date = pd.to_datetime(df['first_join_date'], errors='coerce')
            df['membership_days'] = (pd.Timestamp.now() - join_date).dt.days

        if 'tax_bracket' in df.columns and 'tax_rate' not in df.columns:
            if isinstance(df['tax_bracket'], str):
                df['tax_rate'] = df['tax_bracket'].apply(lambda x: float(x.replace('%','')) / 100)
            else:
                df['tax_rate'] = df['tax_bracket']

        if 'transaction_stability_index' in df.columns and 'TSI' not in df.columns:
            df['TSI'] = df['transaction_stability_index']


        # Drop columns not needed in inference
        drop_cols = [
            'customer_id', 'first_join_date', 'total_transactions',
            'avg_monthly_transaction_count', 'demographic_score',
            'customer_segment', 'tax_bracket', 'transaction_stability_index',
            'credit_card', 'direct_debit'
        ]
        drop_cols.extend(self.labels_col)
        
        cols_to_drop = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=cols_to_drop)

        # Double check the columns for inference
        required_cols = [
            'residence_country', 'gender', 'age', 'residence_index',
            'channel_entrace', 'activity_status', 'household_gross_income',
            'personal_income', 'number_of_children', 'employment_status',
            'current_loan_amount', 'credit_score', 'min_balance',
            'max_balance', 'avg_balance', 'tax_rate', 'avg_income_days_per_month',
            'income_amount_cv', 'avg_expense_days_per_month', 'expense_amount_cv',
            'avg_transactions_per_month', 'monthly_transaction_std', 'active_months',
            'SPS', 'TSI', 'membership_days'
        ]
        for col in required_cols:
            if col not in X.columns:
                print(f"Adding missing column: {col} with default 0")
                X[col] = 0

        # Define a new list with the defined order
        X = X[required_cols]

        return X


    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please load the model first.")
        
        return self.model.predict_proba(X)


    def recommend_products(self, customer_df, current_products=None, top_n=3):
        if self.model is None:
            return ValueError("Model is not loaded. Please load the model first.")
        
        try:
            X = self.preprocess_data(customer_df)
            probas = self.predict_proba(X)
            probs_matrix = np.column_stack([p[:, 1] for p in probas])
            
            probs_df = pd.DataFrame(probs_matrix, columns=self.labels_col, index=X.index)

            # Masking owned products
            if current_products is not None:
                for product in self.labels_col:
                    if product in current_products and current_products[product].iloc[0] == 1:
                        probs_df[product] = -1

            recommendations = []

            for product, prob in probs_df.iloc[0].sort_values(ascending=False).items():
                if prob > 0:
                    recommendations.append(product)
                    if len(recommendations) >= top_n:
                        break

            result = {
                'recommended_products': recommendations
            }

            return result
        except Exception as e:
            print(f"Error dung model inference: {e}")


def main():

    # # Get the input for testing
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # customers_path = os.path.join(script_dir, "../../data/demo_data/customers_demo.csv")
    # transactions_path = os.path.join(script_dir, "../../data/demo_data/transactions_demo.csv")
    # customers = pd.read_csv(customers_path)
    # transactions = pd.read_csv(transactions_path)
    # customer_id = 1405924
    # customer_df = customers[customers['customer_id'] == customer_id]
    # transactions_df = transactions[transactions['customer_id'] == customer_id]
    # feature_engineer = FeatureEngineer()
    # features, customer_df_copy = feature_engineer.generate_features(customer_df, transactions_df)

    # # Testing current module
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(script_dir, "../../models/primary_model/best_multilabel_model.pkl")
    # model = ProductCategoryRecommender(model_path)

    # recommendations = model.recommend_products(customer_df_copy)
    # print(recommendations)
    pass

if __name__ == "__main__":
    main()