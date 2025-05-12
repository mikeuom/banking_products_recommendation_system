import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Initiate the Loan Product Recommender class, utlizing the rule-based logic that was used to generate the loan products recommendations
# The same logic of the model is provided in the notebook file, refer to folder loans_products_prediction_model in notebooks_models_development

class LoanProductRecommender:

    def __init__(self):
        self.loan_products = {
            'personal_loan': {
                'name': 'Personal Loan',
                'interest_rate_range': (4.35, 8.50),
                'term_range': (1, 5),
                'min_credit_score': 580,
                'max_dti': 0.50
            },
            'home_improvement_loan': {
                'name': 'Home Improvement Loan',
                'interest_rate_range': (4.50, 7.25),
                'term_range': (1, 5),
                'min_credit_score': 600,
                'max_dti': 0.45
            },
            'auto_loan': {
                'name': 'Auto Loan',
                'interest_rate_range': (3.85, 6.75),
                'term_range': (1, 5),
                'min_credit_score': 560,
                'max_dti': 0.50
            },
            'education_loan': {
                'name': 'Education Loan',
                'interest_rate_range': (4.05, 6.15),
                'term_range': (1, 10),
                'min_credit_score': 600,
                'max_dti': 0.45
            },
            'travel_loan': {
                'name': 'Travel Loan',
                'interest_rate_range': (4.75, 9.00),
                'term_range': (1, 3),
                'min_credit_score': 600,
                'max_dti': 0.45
            },
            'debt_consolidation_loan': {
                'name': 'Debt Consolidation Loan',
                'interest_rate_range': (4.75, 8.50),
                'term_range': (1, 5),
                'min_credit_score': 580,
                'max_dti': 0.60
            },
            'business_loan': {
                'name': 'Small Business Loan',
                'interest_rate_range': (4.35, 9.50),
                'term_range': (1, 5),
                'min_credit_score': 620,
                'max_dti': 0.45
            },
            'medical_loan': {
                'name': 'Medical Loan',
                'interest_rate_range': (4.15, 7.50),
                'term_range': (1, 5),
                'min_credit_score': 580,
                'max_dti': 0.50
            },
            'no_loan': {
                'name': 'No Loan',
                'interest_rate_range': (0, 0),
                'term_range': (0, 0),
                'min_credit_score': 0,
                'max_dti': 0
            }
        }
        
        self.category_to_loan = {
            'HOME': 'home_improvement_loan',
            'AUTO': 'auto_loan',
            'RETAIL': 'personal_loan',
            'TRAVEL': 'travel_loan',
            'EDUCATION': 'education_loan',
            'HEALTH': 'medical_loan',
            'BUSINESS': 'business_loan',
            'OTHER': 'personal_loan'  # Default to personal loan
        }
        
        # MCC to category mapping for transaction analysis
        self.mcc_to_category = {
            # HOME
            5211: 'HOME', 1711: 'HOME', 5712: 'HOME', 5719: 'HOME', 3174: 'HOME', 
            3144: 'HOME', 5722: 'HOME', 3260: 'HOME', 5251: 'HOME', 3640: 'HOME', 
            3007: 'HOME', 3005: 'HOME', 3009: 'HOME', 3256: 'HOME',
            
            # AUTO
            5541: 'AUTO', 7538: 'AUTO', 5533: 'AUTO', 7531: 'AUTO', 7542: 'AUTO', 
            7549: 'AUTO', 4784: 'AUTO', 4121: 'AUTO',
            
            # RETAIL
            5311: 'RETAIL', 5661: 'RETAIL', 5651: 'RETAIL', 5621: 'RETAIL', 5732: 'RETAIL', 
            5310: 'RETAIL', 5977: 'RETAIL', 5655: 'RETAIL', 5941: 'RETAIL', 5733: 'RETAIL', 
            5947: 'RETAIL', 5970: 'RETAIL', 5932: 'RETAIL', 5094: 'RETAIL', 5816: 'RETAIL', 
            5815: 'RETAIL', 5300: 'RETAIL', 5411: 'RETAIL', 5499: 'RETAIL', 5912: 'RETAIL',
            
            # TRAVEL
            4722: 'TRAVEL', 7011: 'TRAVEL', 4511: 'TRAVEL', 4111: 'TRAVEL', 3722: 'TRAVEL', 
            4112: 'TRAVEL', 4131: 'TRAVEL', 4411: 'TRAVEL', 3771: 'TRAVEL', 3775: 'TRAVEL',
            
            # EDUCATION
            5942: 'EDUCATION', 5192: 'EDUCATION', 8931: 'EDUCATION', 7276: 'EDUCATION',
            
            # HEALTH
            8099: 'HEALTH', 8021: 'HEALTH', 8011: 'HEALTH', 8041: 'HEALTH', 8043: 'HEALTH', 
            8049: 'HEALTH', 8062: 'HEALTH', 7230: 'HEALTH',
            
            # BUSINESS
            8111: 'BUSINESS', 7349: 'BUSINESS', 7393: 'BUSINESS', 7210: 'BUSINESS', 3780: 'BUSINESS', 
            5045: 'BUSINESS', 6300: 'BUSINESS', 4214: 'BUSINESS', 3509: 'BUSINESS', 9402: 'BUSINESS', 
            3390: 'BUSINESS', 3596: 'BUSINESS', 3730: 'BUSINESS', 3684: 'BUSINESS', 3504: 'BUSINESS', 
            3389: 'BUSINESS', 3393: 'BUSINESS', 3395: 'BUSINESS', 3058: 'BUSINESS', 3387: 'BUSINESS', 
            3405: 'BUSINESS', 3132: 'BUSINESS', 3359: 'BUSINESS', 3000: 'BUSINESS', 3001: 'BUSINESS', 
            3006: 'BUSINESS', 3008: 'BUSINESS', 3075: 'BUSINESS', 3066: 'BUSINESS'
        }
        
        # Minimum spending thresholds
        self.min_spend_thresholds = {
            'HOME': 50,
            'AUTO': 50,
            'RETAIL': 25,
            'TRAVEL': 50,
            'EDUCATION': 50,
            'HEALTH': 50,
            'BUSINESS': 100,
            'OTHER': 10
        }

    def process_transactions(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        if transaction_df.empty:
            return pd.DataFrame()
            
        transaction_df = transaction_df.copy()
        
        if 'amount' in transaction_df.columns:
            if transaction_df['amount'].dtype == 'object':
                transaction_df['amount'] = transaction_df['amount'].str.replace('[$,)]', '', regex=True)
                transaction_df['amount'] = transaction_df['amount'].str.replace('[(]', '-', regex=True)
                transaction_df['amount'] = pd.to_numeric(transaction_df['amount'], errors='coerce')
            
            transaction_df['amount_abs'] = transaction_df['amount'].abs()
        elif 'amount_abs' not in transaction_df.columns:
            return pd.DataFrame()
        
        # Filter recent transactions
        if 'date' in transaction_df.columns:
            transaction_df['date'] = pd.to_datetime(transaction_df['date'], errors='coerce')
            latest_date = transaction_df['date'].max()
            start_date = latest_date - pd.Timedelta(days=120)
            transaction_df = transaction_df[transaction_df['date'] >= start_date]
        
        # Keep necessary columns
        if 'customer_id' in transaction_df.columns and 'mcc' in transaction_df.columns:
            transaction_df = transaction_df[['customer_id', 'mcc', 'amount_abs']]
        else:
            return pd.DataFrame()
        
        # Map MCC
        mcc_series = pd.Series(self.mcc_to_category)
        transaction_df['category'] = transaction_df['mcc'].map(mcc_series)
        transaction_df['category'] = transaction_df['category'].fillna('OTHER')
        
        # Calculate spending statistics
        spending_pivot = transaction_df.pivot_table(
            index='customer_id',
            columns='category',
            values='amount_abs',
            aggfunc='sum',
            fill_value=0
        )
        
        all_categories = ['HOME', 'AUTO', 'RETAIL', 'TRAVEL', 'EDUCATION', 'HEALTH', 'BUSINESS', 'OTHER']
        for category in all_categories:
            if category not in spending_pivot.columns:
                spending_pivot[category] = 0
        
        spending_pivot['TOTAL'] = spending_pivot[all_categories].sum(axis=1)
        
        spending_pivot['dominant_category'] = spending_pivot[all_categories].idxmax(axis=1)
        
        spending_pivot['dominant_amount'] = spending_pivot.apply(
            lambda row: row[row['dominant_category']] if row['dominant_category'] in all_categories else 0,
            axis=1
        )
        
        return spending_pivot

    def get_interest_rate(self, loan_product: str, credit_score: float) -> Optional[float]:
        if loan_product == 'no_loan':
            return None
            
        if loan_product not in self.loan_products:
            return None
            
        min_rate, max_rate = self.loan_products[loan_product]['interest_rate_range']
        
        # Better credit score = lower interest rate
        score_ratio = min(1, max(0, (credit_score - 560) / 240))
        interest_rate = max_rate - score_ratio * (max_rate - min_rate)
        
        return round(interest_rate, 2)

    def recommend_loan_products(self, 
                              customer_df: pd.DataFrame, 
                              transaction_df: pd.DataFrame) -> Dict:
        if customer_df.empty:
            return {
                'is_eligible': False,
                'recommended_product': 'no_loan',
                'loan_name': 'No Loan',
                'interest_rate': None,
                'min_term': None,
                'max_term': None,
                'reason': 'No customer data available'
            }
        
        spending_data = self.process_transactions(transaction_df)
        
        if spending_data.empty:
            customer_id = customer_df['customer_id'].iloc[0] if 'customer_id' in customer_df.columns else 0
            spending_data = pd.DataFrame({
                'HOME': [0], 'AUTO': [0], 'RETAIL': [0], 'TRAVEL': [0], 
                'EDUCATION': [0], 'HEALTH': [0], 'BUSINESS': [0], 'OTHER': [0],
                'TOTAL': [0], 'dominant_category': ['OTHER'], 'dominant_amount': [0]
            }, index=[customer_id])
            
        # credit score and dti_ratio
        credit_score = 650
        dti_ratio = 0.3
        
        if 'credit_score' in customer_df.columns:
            credit_score = customer_df['credit_score'].iloc[0]
            if pd.isna(credit_score):
                credit_score = 650
        
        if 'personal_income' in customer_df.columns and 'current_loan_amount' in customer_df.columns:
            income = customer_df['personal_income'].iloc[0]
            current_loan = customer_df['current_loan_amount'].iloc[0]
            
            if not pd.isna(income) and not pd.isna(current_loan) and income > 0:
                dti_ratio = current_loan / income
            
        recommendation = {
            'is_eligible': False,
            'recommended_product': 'no_loan',
            'loan_name': 'No Loan',
            'interest_rate': None,
            'min_term': None,
            'max_term': None,
            'reason': 'Not eligible for any loan product'
        }
        
        # Get dominant category
        dominant_category = spending_data['dominant_category'].iloc[0]
        dominant_amount = spending_data['dominant_amount'].iloc[0]
        
        # Check eligibility based on spending
        if dominant_amount >= self.min_spend_thresholds.get(dominant_category, 10):
            loan_product = self.category_to_loan.get(dominant_category, 'personal_loan')
            product_criteria = self.loan_products[loan_product]
            
            # Check if customer meets criteria
            if (credit_score >= product_criteria['min_credit_score'] - 20 and 
                dti_ratio <= product_criteria['max_dti'] + 0.05):
                
                recommendation.update({
                    'is_eligible': True,
                    'recommended_product': loan_product,
                    'loan_name': product_criteria['name'],
                    'interest_rate': self.get_interest_rate(loan_product, credit_score),
                    'min_term': product_criteria['term_range'][0],
                    'max_term': product_criteria['term_range'][1],
                    'reason': f"Based on the customer's dominant spending in {dominant_category.lower()}"
                })
                
                return recommendation
        
        # Check debt consolidation as alternative
        if dti_ratio > 0.25 and credit_score >= self.loan_products['debt_consolidation_loan']['min_credit_score'] - 20:
            recommendation.update({
                'is_eligible': True,
                'recommended_product': 'debt_consolidation_loan',
                'loan_name': self.loan_products['debt_consolidation_loan']['name'],
                'interest_rate': self.get_interest_rate('debt_consolidation_loan', credit_score),
                'min_term': self.loan_products['debt_consolidation_loan']['term_range'][0],
                'max_term': self.loan_products['debt_consolidation_loan']['term_range'][1],
                'reason': "Based on the customer's debt profile"
            })
            
            return recommendation
            
        # Last resort: personal loan with minimal criteria
        if credit_score >= 540:
            recommendation.update({
                'is_eligible': True,
                'recommended_product': 'personal_loan',
                'loan_name': self.loan_products['personal_loan']['name'],
                'interest_rate': self.get_interest_rate('personal_loan', credit_score),
                'min_term': self.loan_products['personal_loan']['term_range'][0],
                'max_term': self.loan_products['personal_loan']['term_range'][1],
                'reason': "Based on the customer's overall profile"
            })
            
            return recommendation
            
        return recommendation