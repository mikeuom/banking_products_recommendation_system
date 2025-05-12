import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initiate the Feature Engineer class, utlizing the logic that was used to generate new computed features for the customers' profiles dataset
# The same logic of the model is provided in the notebook file, refer to folder feature_engineering in notebooks_models_development

class FeatureEngineer:

    def __init__(self):
        self.features = {}


    def assign_tax_bracket(self, personal_income):
        if personal_income <= 1047.5:
            return '0%'
        elif personal_income <= 4189.2:
            return '20%'
        elif personal_income <= 10428.3:
            return '40%'
        else:
            return '45%'

    def get_tax_rate(self, personal_income):
        if personal_income <= 1047.5:
            return 0.0
        elif personal_income <= 4189.2:
            return 0.2
        elif personal_income <= 10428.3:
            return 0.4
        else:
            return 0.45


    def calculate_income_stability(self, transactions_df):

        income_df = transactions_df[transactions_df['transaction_type'] == 'Deposit'].copy()
        if len(income_df) == 0:
            return {
                'avg_income_days_per_month': 0,
                'income_amount_cv': 0
            }
        
        income_df['date'] = pd.to_datetime(income_df['date'])
        income_df['month'] = income_df['date'].dt.to_period('M')
        income_df['week'] = income_df['date'].dt.to_period('W')
        income_df['day'] = income_df['date'].dt.date

        # Average days of income per month - the average number of days where income is recorded
        daily_income = income_df.groupby(['customer_id', 'month', 'day'])['amount'].sum().reset_index()
        days_per_month = daily_income.groupby(['customer_id', 'month']).size().reset_index(name='income_days')
        avg_days = float(days_per_month['income_days'].mean())

        # CV(Coefficient of Variation) of monthly income amounts - varitation of income from month to month
        monthly_total = income_df.groupby(['customer_id', 'month'])['amount'].sum().reset_index()

        if len(monthly_total) > 1:
            monthly_cv = round(float(monthly_total['amount'].std() / monthly_total['amount'].mean()), 2)
        else:
            monthly_cv = 0

        return {
            'avg_income_days_per_month': avg_days,
            'income_amount_cv': monthly_cv
        }
    

    def calculate_expense_regularity(self, transactions_df):
        expense_df = transactions_df[transactions_df['transaction_type'] == 'Withdrawal'].copy()
        if len(expense_df) == 0:
            return {
                'avg_expense_days_per_month': 0,
                'expense_amount_cv': 0
            }
        
        expense_df['date'] = pd.to_datetime(expense_df['date'])
        expense_df['month'] = expense_df['date'].dt.to_period('M')
        expense_df['day'] = expense_df['date'].dt.date

        # Average days of expense per month - the average number of days where expenses are recorded
        daily_expense = expense_df.groupby(['customer_id', 'month', 'day'])['amount'].sum().reset_index()
        days_per_month = daily_expense.groupby(['customer_id', 'month']).size().reset_index(name='expense_days')
        avg_days = round(float(days_per_month['expense_days'].mean()), 2)

        # CV(Coefficient of Variation) of monthly expense amounts - varitation of expenses from month to month
        monthly_total = expense_df.groupby(['customer_id', 'month'])['amount'].sum().reset_index()
        monthly_amount_std = float(monthly_total['amount'].std())
        monthly_amount_mean = float(monthly_total['amount'].mean())
        monthly_cv = round(monthly_amount_std / monthly_amount_mean, 2)

        return {
            'avg_expense_days_per_month': avg_days,
            'expense_amount_cv': monthly_cv
        }

    def calculate_transaction_volume(self, transactions_df):
        df = transactions_df.copy()
        if len(df) == 0:
            return {
                'avg_transactions_per_month': 0,
                'monthly_transaction_std': 0,
                'total_transactions': 0,
                'active_months': 0,
                'avg_monthly_transaction_count': 0
            }

        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')

        monthly_tx = df.groupby(['customer_id', 'year_month']).size().reset_index(name='tx_count')
        
        avg_tx = round(float(monthly_tx['tx_count'].mean()), 2)
        std_tx = round(float(monthly_tx['tx_count'].std()), 2)
        total_tx = len(df)
        active_months = len(monthly_tx)
        avg_monthly_tx = round(float(total_tx/active_months), 2)

        return {
            'avg_transactions_per_month': avg_tx,
            'monthly_transaction_std': std_tx,
            'total_transactions': total_tx,
            'active_months': active_months,
            'avg_monthly_transaction_count': avg_monthly_tx
        }
    
    # For inference stage: working with Scalar values instead of DataFrame
    def calculate_sps(self, customer_df, transactions_df):
        normalized_age = float((customer_df['age'].iloc[0] - 18) / (70 - 18))
        normalized_income = float(customer_df['personal_income'].iloc[0] / 100000)

        tenure_score_map = {
            "0-1 year": 0.3,
            "2-4 years": 0.6,
            "More than 5 years": 1.0
        }
        normalized_tenure = tenure_score_map.get(customer_df['customer_segment'].iloc[0], 0.3)

        normalized_balance = float(customer_df['avg_balance'].iloc[0] / 50000)

        deposits = float(transactions_df[transactions_df['transaction_type'] == 'Deposit']['amount'].sum())
        withdrawals = float(transactions_df[transactions_df['transaction_type'] != 'Deposit']['amount'].abs().sum())
        net_cash_flow = deposits - withdrawals
        normalized_cash_flow = net_cash_flow / 10000

        deposit_freq = len(transactions_df[transactions_df['transaction_type'] == 'Deposit'])
        withdrawal_freq = len(transactions_df[transactions_df['transaction_type'] != 'Deposit'])
        normalized_deposit_freq = deposit_freq / 20
        normalized_withdrawal_freq = withdrawal_freq / 20

        # Calculate SPS
        w1, w2, w3, w4, w5, w6 = 0.2, 0.2, 0.2, 0.2, 0.1, 0.1
        sps = (
                w1 * normalized_balance +
                w2 * normalized_cash_flow +
                w3 * normalized_deposit_freq +
                w4 * normalized_withdrawal_freq +
                w5 * normalized_tenure +
                w6 * normalized_age
        )

        return {'SPS': round(float(max(0, min(1, sps))), 2)}
    

    def calculate_transaction_stability_index(self, transactions_df):
        if len(transactions_df) == 0:
            return {
                'transaction_stability_index': 0,
                'TSI': 0
            }

        deposit_df = transactions_df[transactions_df['transaction_type'] == 'Deposit'].copy()
        withdrawals = transactions_df[transactions_df['transaction_type'] == 'Withdrawal'].copy()   

        if len(deposit_df) > 1:
            cv_deposits = float(deposit_df['amount'].std() / deposit_df['amount'].mean())
        else:
            cv_deposits = 0

        if len(withdrawals) > 1:
            cv_withdrawals = float(withdrawals['amount'].std() / withdrawals['amount'].mean())
        else:
            cv_withdrawals = 0
        

        tsi = abs(1 - (cv_deposits + cv_withdrawals) / 2)
        tsi_value = round(float(tsi), 2)

        return {
            'transaction_stability_index': tsi_value,
            'TSI': tsi_value
        }

    def calculate_demographic_score(self, customer_df):

        normalized_age = (customer_df['age'] - 18) / (70 - 18)

        normalized_income = customer_df['personal_income'] / 100000

        employment_factor = customer_df['employment_status']

        tenure_score_map = {
            "0-1 year": 0.3,
            "2-4 years": 0.6,
            "More than 5 years": 1.0
        }
        tenure_factor = tenure_score_map.get(customer_df['customer_segment'].iloc[0], 0.3)

        demographic_score = round(float((normalized_age + normalized_income + employment_factor + tenure_factor) / 4), 2)

        return demographic_score


    def generate_features(self, customer_df, transactions_df):        
        # Tax bracket and rate
        personal_income = customer_df['personal_income'].values[0]
        self.features['tax_bracket'] = self.assign_tax_bracket(personal_income)
        self.features['tax_rate'] = self.get_tax_rate(personal_income)

        # Income stability
        income_stability = self.calculate_income_stability(transactions_df)
        self.features.update(income_stability)

        # Expense regularity
        expense_regularity = self.calculate_expense_regularity(transactions_df)
        self.features.update(expense_regularity)

        # Transaction volume
        transaction_volume = self.calculate_transaction_volume(transactions_df)
        self.features.update(transaction_volume)

        # SPS - Saving Prospensity Score
        sps = self.calculate_sps(customer_df, transactions_df)
        self.features.update(sps)

        # Transaction Stability Index
        transaction_stability = self.calculate_transaction_stability_index(transactions_df)
        self.features.update(transaction_stability)

        # Demographic Score
        demographic_score = self.calculate_demographic_score(customer_df)
        self.features.update({'demographic_score': demographic_score})

        customer_df_copy = customer_df.copy()
        for key, value in self.features.items():
            customer_df_copy[key] = value

        return self.features, customer_df_copy


def main():
    customers = pd.read_csv("../data/demo_data/customers_demo.csv")
    transactions = pd.read_csv("../data/demo_data/transactions_demo.csv")

    customer_id = 1405924
    customer_df = customers[customers['customer_id'] == customer_id]
    transactions_df = transactions[transactions['customer_id'] == customer_id]

    feature_engineer = FeatureEngineer()
    features, customer_df_copy = feature_engineer.generate_features(customer_df, transactions_df)

    return features, customer_df_copy

if __name__ == "__main__":
    main()