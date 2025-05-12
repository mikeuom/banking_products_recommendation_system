import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os
import webbrowser
from pathlib import Path
import base64

# Loading all the components of the recommendation pipeline: Feature engineering module, Predictive models and a rule-based model
from utils.loan_product_recommender import LoanProductRecommender
from utils.feature_engineering import FeatureEngineer
from utils.product_category_recommender import ProductCategoryRecommender
from utils.customer_segmenter import CustomerSegmenter
from utils.credit_card_recommender import CreditCardRecommender

'''
This is the main main user interface of the retail banking products recommendation pipeline
To run the app, run the following command in the terminal: streamlit run main.py
'''

def get_current_products(customer_df):
    product_columns = ['saving_account', 'guarantees', 'junior_account', 'loans', 'pension', 'credit_card', 'direct_debit']
    current_products = {}
    
    for col in product_columns:
        if col in customer_df.columns:
            current_products[col] = customer_df[col].iloc[0]
        else:
            current_products[col] = 0
            
    return current_products

def get_html_link(html_path, link_text):
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'data:text/html;base64,{b64}'
    return f'<a href="{href}" target="_blank">{link_text}</a>'


vertical_pipeline = Image.open("assets/vertical_pipeline.jpg")

st.set_page_config(
    page_title="Banking Products Recommendation",
    page_icon="💰",
    layout="wide"
)

@st.cache_data
def fetch_images():
    vertical_pipeline = Image.open("assets/vertical_pipeline.jpg")
    return vertical_pipeline



# Fetch the data from the csv files in the data/background_data folder - the data is used for the demo particularly
@st.cache_data
def fetch_data():
    customers = pd.read_csv("../data/background_data/customer.csv")
    transactions_count = pd.read_csv("../data/background_data/transaction_count.csv")
    cus_segments = pd.read_csv("../data/background_data/transaction_data_recommendations.csv")
    loan_recommendations = pd.read_csv("../data/background_data/loan_recommendation_with_buying_indicator.csv")
    credit_card_recommendations = pd.read_csv("../data/background_data/credit_card_recommendation_with_buying_indicator.csv")
    return customers, transactions_count, cus_segments, loan_recommendations, credit_card_recommendations


vertical_pipeline = fetch_images()
customers, transactions_count, cus_segments, loan_recommendations, credit_card_recommendations = fetch_data()

cus_segments = cus_segments[
    (cus_segments['recency'] < cus_segments['recency'].quantile(0.99)) &
    (cus_segments['monetary'] < cus_segments['monetary'].quantile(0.99)) &
    (cus_segments['frequency'] < cus_segments['frequency'].quantile(0.99))
]

st.title("Banking Products Recommendation Pipeline for Retail Customers")

# Define the tabs in the main user interface
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Introduction",
    "Prediction Pipeline Description",
    "Product Recommendation Pipeline Demo",
    "Current Data Overview",
    "Loan Products Recommendations",
    "Credit Card Products Recommendations"
])

# INTRODUCTION TAB
with tab1:
    st.write("""
     Welcome to the Banking Products Recommendation Pipeline. Our pipeline is designed to recommend the most suitable
             banking products to the retail customers based on their profile and behaviors.

    Following are the main sections in this page:
    1. **Introduction**
    2. **Prediction Pipeline Description** - Summary of the prediction pipeline
    3. **Product Recommendation Pipeline Demo** - Demonstration of the product recommendation pipeline
    4. **Current Data Overview** - Analysis of the current product ownership and customer segmentation
    5. **Loan Products Recommendations** - Summary of the loan products recommendations results
    6. **Credit Card Products Recommendations** - Summary of the credit card products recommendations results

    The data sources is derived from customer profiles and transactional records, providing a comprehensive view of customer behavior and product preferences.
    """)

# PREDICTION PIPELINE DESCRIPTION TAB
with tab2:
    st.write("""Our pipeline demonstrates how banks can leverage different data types (profile vs. transactional) to create retail banking product recommendations 
             through machine learning and data segmentation techniques. There are two main pathways in our recommendation system:
             """)
    
    st.write("""
     The first pathway recommends savings accounts, guarantees, junior accounts, loans, and pension products based on customer profile data. 
              The process begins with demographic information, credit scores, and income data, which feeds into a machine learning model that 
              predicts appropriate product categories. This two-stage approach first determines suitable product categories 
              before refining specific product recommendations.
              """)
    
    st.write("""
            The second pathway focuses on recommending direct debit and credit card products using transactional history. 
            The process starts with analyzing historical purchase data and spending patterns, then extracts RFM (Recency, Frequency, Monetary value) metrics. 
            Customers are segmented into clusters (high-value loyalists, occasional big-spenders, and regular modest customers) using clustering algorithms. 
            The final predictive model then recommends specific credit card products based on these segments. 
            The example shows Customer 122324 being recommended a Retail Credit Card after being analyzed through this three-stage process.
            """)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(vertical_pipeline, width=800)

# CURRENT DATA OVERVIEW TAB
with tab4:
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.subheader("Product Ownership Distribution")
        product_list = ["Savings Account", "Guarantees", "Junior Account", "Loans", "Credit Card",
                        "Pension", "Direct Debit"]
        product_columns = ['saving_account', 'guarantees', 'junior_account', 'loans', 'credit_card', 
                        'pension', 'direct_debit']
        columns = product_list
        counts = customers[product_columns].sum()
        fig_product_dist = px.bar(
            x=product_list,
            y=counts,
            labels={'x': 'Product Type', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig_product_dist, use_container_width=True, key="fig_product_dist")

        st.subheader("Customer Group Distribution")
        segment_counts = customers['customer_segment'].value_counts()
        fig_customer_group = px.pie(
            values= segment_counts.values,
            names= segment_counts.index
        )
        st.plotly_chart(fig_customer_group, use_container_width=True, key="fig_customer_group")


    with col2:
        st.subheader("Customer Segmentation")
        fig_customer_seg = go.Figure(data=[go.Scatter3d(
                x=cus_segments['recency'],
                y=cus_segments['monetary'],
                z=cus_segments['frequency'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=cus_segments['cluster'],
                    colorscale="Viridis",
                    opacity=0.8
                ),
                text=cus_segments['customer_id'],
                hovertemplate='<b>Customer ID:</b> %{text}<br>' +
                            '<b>Recency:</b> %{x}<br>' +
                            '<b>Monetary:</b> %{y}<br>' +
                            '<b>Frequency:</b> %{z}<br>' +
                            '<b>Cluster:</b> %{marker.color}<extra></extra>'
            )])

        fig_customer_seg.update_layout(
            scene=dict(
                xaxis_title = "Recency",
                yaxis_title = "Monetary",
                zaxis_title = "Frequency",
                dragmode=False,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=1000,
            margin = dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )   
        st.plotly_chart(fig_customer_seg, use_container_width=True, key="fig_customer_seg")

    st.subheader("Recent amount of transactions in the last 12 months")
    fig_trans_count = px.line(
        transactions_count,
        x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        y='transaction_count'
    )
    fig_trans_count.update_layout(
        xaxis_title='Month',
        yaxis_title='Transaction Count',
        xaxis_tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        xaxis_ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    st.plotly_chart(fig_trans_count, use_container_width=True, key="fig_trans_count")

# LOAN PRODUCT TAB
with tab5:
    st.subheader("Loan Product Recommendations results")
    st.write("""
             The loan product recommendation is based on the customer's profile and transactional data.
             """)
    
    container1, container2, container3, container4 = st.columns(4)
    with container1:
        with st.container(border=True):
            total_customers = len(customers)
            total_customers_loans = len(customers[customers['loans'] == 1])
            st.metric("Customers having Loans", total_customers_loans)
    with container2:
        with st.container(border=True):
            total_customers = len(customers)
            total_customers_loans = len(customers[customers['loans'] == 1])
            percentage_customers_loans = round(total_customers_loans/total_customers * 100, 2)
            st.metric("Percentage of customers having loans", f"{percentage_customers_loans}%")
    with container3:
        with st.container(border=True):
            mask = loan_recommendations['recommended_product'] != 'no_loan'
            num_recommended = len(loan_recommendations[mask])
            st.metric("Customers recommended for loan products", num_recommended)
    with container4:
        with st.container(border=True):
            num_bought = loan_recommendations['buying_indicator'].sum()
            conversion_rate_loan = round(num_bought/num_recommended * 100, 2)
            st.metric("Current conversion rate of the recommendations", f"{conversion_rate_loan}%")

    col1, col2 = st.columns(2)
    with col1:
        product_counts = loan_recommendations[mask]['recommended_product'].value_counts()
        fig_loan_dist = px.bar(
            x=product_counts.values,
            y=product_counts.index,
            orientation='h',
            title="Distribution of Recommended Loan Products",
            labels={'x': 'Number of Recommendations', 'y': 'Loan Product'}
        )
        st.plotly_chart(fig_loan_dist, use_container_width=True, key="fig_loan_dist")
    
    with col2:
        eligible = len(loan_recommendations[loan_recommendations['recommended_product'] != 'no_loan'])
        not_eligible = len(loan_recommendations[loan_recommendations['recommended_product'] == 'no_loan'])
        fig_eligibility = px.pie(
            values=[eligible, not_eligible],
            names=['Eligible', 'Not Eligible'],
            title="Customer Eligibility Distribution",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig_eligibility, use_container_width=True, key="fig_eligibility")

    with st.expander("View the current list of customers being offered"):
        st.dataframe(loan_recommendations)

# CREDIT CARD PRODUCT TAB
with tab6:
    st.header("Credit Card Product Recommendations results")
    st.write("""
             The credit card product recommendation is mostly based on the customer's historical transactional data.
             """)
    
    total_customers = len(customers)
    total_customers_credit_card = len(customers[customers['credit_card'] == 1])
    percentage_customers_credit_card = round(total_customers_credit_card/total_customers * 100, 2)
    credit_card_mask = credit_card_recommendations['final_offer'] != 'No Targeted Offer'
    customers_rec_credit_card = credit_card_recommendations[credit_card_mask]
    conversion_rate_credit_card = round(customers_rec_credit_card['buying_indicator'].sum()/len(customers_rec_credit_card) * 100, 2)

    container1, container2, container3, container4 = st.columns(4)
    with container1:
        with st.container(border=True):
            st.metric("Customers having Credit Card", total_customers_credit_card)
    with container2:
        with st.container(border=True):
            st.metric("Percentage of customers having credit card", f"{percentage_customers_credit_card}%")
    with container3:
        with st.container(border=True):
            st.metric("Customers recommended for credit card products", len(customers_rec_credit_card))
    with container4:
        with st.container(border=True):
            st.metric("Current conversion rate of the recommendations", f"{conversion_rate_credit_card}%")        
    
    cred_product_counts = customers_rec_credit_card['final_offer'].value_counts()
    clusters_product_counts = customers_rec_credit_card.groupby(['cluster']).size().reset_index(name='count')
    clusters_product_counts = clusters_product_counts.sort_values(by='cluster', ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        treemap_data = pd.DataFrame({
            'Credit Card Type': cred_product_counts.index,
            'Count': cred_product_counts.values
        })
        
        fig_credit_card_treemap = px.treemap(
            treemap_data,
            path=['Credit Card Type'],
            values='Count',
            title="Distribution of Recommended Credit Cards (Treemap)"
        )
        st.plotly_chart(fig_credit_card_treemap, use_container_width=True, key="fig_credit_card_treemap")


    with col2:
        fig = go.Figure(
            data=[go.Pie(
                labels=cred_product_counts.index,
                values=cred_product_counts.values,
                hole=0.6,
                hoverinfo='label+percent+value',
                marker=dict(colors=px.colors.qualitative.Pastel)
            )]
        )
        fig.update_layout(title_text='Final Offer Distribution (Interactive Donut)')
        st.plotly_chart(fig, use_container_width=True, key="fig_donut")

    with st.expander("View the current list of customers being offered"):
        st.dataframe(credit_card_recommendations)

    html_file_path = "assets/pca_4d_interactive.html"
    if os.path.exists(html_file_path):
        file_path = os.path.abspath(html_file_path) 
        viz_tab1, viz_tab2 = st.tabs(["Embedded View", "External View"])
        
        with viz_tab1:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read() 
            
            # Create a full-width container for the visualization
            with st.container():
                st.components.v1.html(html_content, height=3200, width=None, scrolling=False)
        
        with viz_tab2:
            st.write("### Open Visualization in New Tab")
            st.write("Click the button below to open the visualization in a new browser tab:")
            
            if st.button("Open Full Screen in Browser", type="primary"):
                # Open file in the web browser
                webbrowser.open(f'file://{file_path}', new=2)
                st.success("Visualization opened in a new browser tab")
    else:
        st.error(f"Visualization file not found: {html_file_path}")

# PRODUCT RECOMMENDATION DEMO TAB - THIS IS THE HEART OF THE PROTOTOTYPE, SHOWING THE RECOMMENDATION PIPELINE IN ACTION
with tab3:
    st.write("""
             This section demonstrates the product recommendation pipeline in action.
             The internal staffs or sales team can use this pipeline to recommend 
             the most suitable products to the customers.

             The demo will be conducted on a few customers.
             """)
    
    customers_demo = pd.read_csv("../data/demo_data/customers_demo.csv")
    transactions_demo = pd.read_csv("../data/demo_data/transactions_demo.csv")

    st.subheader("1. Please select a customer you want to recommend products to:")
    customer_id = st.selectbox(" ", customers_demo['customer_id'])
    st.write(f"Selected customer: {customer_id}. Following is the customer profile and their recent transactions:")

    # Filtered the customer and transaction data based on the selected customer
    customer_df = customers_demo[customers_demo['customer_id'] == customer_id]
    transactions_df = transactions_demo[transactions_demo['customer_id'] == customer_id]

    with st.expander("View customer profile"):
        st.dataframe(customer_df)
    with st.expander("View recent transactions"):
        st.dataframe(transactions_df)

    st.write("Our pipeline will process the data and generate the following features for the customer:")
    feature_engineer = FeatureEngineer()
    customer_features, customer_df = feature_engineer.generate_features(customer_df, transactions_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Patterns")
        st.metric("Avg Income Days/Month", f"{customer_features['avg_income_days_per_month']:.2f}")
        st.metric("Avg Expense Days/Month", f"{customer_features['avg_expense_days_per_month']:.2f}")
        st.metric("Transactions Per Month", f"{customer_features['avg_monthly_transaction_count']:.2f}")
        st.metric("Active Months", f"{customer_features['active_months']}")
    
    with col2:
        st.subheader("Financial Indicators")
        st.metric("Savings Propensity Score", f"{customer_features['SPS']:.2f}")
        st.metric("Transaction Stability Index", f"{customer_features['transaction_stability_index']:.2f}")
        st.metric("Demographic Score", f"{customer_features['demographic_score']:.2f}")

    with st.expander("View all calculated features"):
        st.dataframe(pd.DataFrame(customer_features, index=[0]))
    
    # First pathway of the recommendation pipeline for savings account, guarantees, junior account, loans, and pension products
    st.subheader("2. Following is the first pathway of the recommendation pipeline for savings account, guarantees, junior account, loans, and pension products:")
    st.write("It will first determine the most suitable product categories for the customer and then predict the product-specific recommendations.")
    
    category_recommender = ProductCategoryRecommender(model_path="../models/primary_model/best_multilabel_model.pkl")

    product_columns = ['saving_account', 'guarantees', 'junior_account', 'loans', 'pension']
    current_products = {
        col: customer_df[col] for col in product_columns
    }
    current_product = pd.DataFrame(current_products)
    
    recommendations = category_recommender.recommend_products(customer_df, current_products)
    
    # Showing the current products that the customer has
    st.write("Current products that the customer has:")
    current_product_list = [product for product in current_product.columns if current_product[product].iloc[0] == 1]
    if current_product_list:
        for product in current_product_list:
            st.info(f"{product.replace('_', ' ').title()} product")
    else:
        st.info("Customer does not have any current products")

    # Showing the recommended products for the customer (savings account, guarantees, junior account, loans, and pension products)
    st.write("Following is the recommended products for the customer:")
    if recommendations['recommended_products']:
        for _, product in enumerate(recommendations['recommended_products']):
            if product == 'loans':
                with st.container(border=True):
                    product_name = product.replace('_', ' ').title()
                    st.info(f"Customer is recommended for {product_name} products.")
                    with st.expander(f"Click here to see the specific {product_name} products recommendation"):
                        # Loan Products Recommender
                        loan_recommender = LoanProductRecommender()
                        loan_recommendation = loan_recommender.recommend_loan_products(customer_df, transactions_df)
                        
                        if loan_recommendation['is_eligible']:
                            st.success(f"Loan Product Offer: **{loan_recommendation['loan_name']}**")
                            
                            # Loan details
                            loan_details = {
                                "Product": loan_recommendation['loan_name'],
                                "Interest Rate": f"{loan_recommendation['interest_rate']}%" if loan_recommendation['interest_rate'] else "N/A",
                                "Term Range": f"{loan_recommendation['min_term']} to {loan_recommendation['max_term']} years" if loan_recommendation['min_term'] and loan_recommendation['max_term'] else "N/A"
                            }
                            
                            st.table(pd.DataFrame([loan_details]).T.rename(columns={0: "Details"}))
                        else:
                            st.warning(f"Not eligible for specific loan products: {loan_recommendation['reason']}")
            else:
                with st.container(border=True):
                    product_name = product.replace('_', ' ').title()
                    st.info(f"Customer is recommended for {product_name} products.")
                    with st.expander(f"Click here to see the specific {product_name} products recommendation"):
                        st.warning(f"A model will be implemented in the future to recommend specific {product_name} products")
    else:
        st.warning("Customer is not recommended for any products")

    # Second pathway of the recommendation pipeline for credit card and direct debit products
    st.subheader("3. This is the second pathway of the recommendation pipeline for credit card and direct debit products:")
    st.write("It will first segment the customer into different clusters based on their transactional data and then recommend the most suitable credit card products.")
    try:
        customer_segmenter = CustomerSegmenter(model_path="../models/customer_segment_model/kmeans_customer_segments.pkl")
        segmentation_results = customer_segmenter.process_and_predict(transactions_df)
        
        segment_data = segmentation_results[segmentation_results['customer_id'] == customer_id].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Customer Segment/Cluster", f"Cluster {segment_data['cluster']}")
            
        with col2:
            st.metric("RFM Score", segment_data['RFM_score'])

        st.write("Recommended products:")
        
        # Initiate credit card model to see if the affinity score is high enough to recommend a credit card
        cc_model_path = "../models/credit_card_model/kmeans_model_creditcardsubtypeunsupervised.pkl"
        credit_card_recommender = CreditCardRecommender(cc_model_path)
        spend_wide = credit_card_recommender.aggregate_spend_wide(transactions_df)
        result = credit_card_recommender.predict_credit_card(spend_wide)

        if segment_data['credit_card_recommendation'] and result['final_offer'].iloc[0] != 'No Targeted Offer':
            with st.container(border=True):
                product_name = 'Credit Card'.replace('_', ' ').title()
                st.info(f"Customer is recommended for Credit Card products.")
                with st.expander(f"Click here to see the specific Credit Card products recommendation"):

                    st.success(f"Credit Card Product Offer: **{result['final_offer'].iloc[0]}**")

                    # Credit Card details
                    credit_card_details = {
                        "Product": result['final_offer'].iloc[0]
                    }

                    st.table(pd.DataFrame([credit_card_details]).T.rename(columns={0: "Details"}))

        if segment_data['direct_debit_recommendation']:
            with st.container(border=True):
                product_name = 'Direct Debit'.replace('_', ' ').title()
                st.info(f"Customer is recommended for Direct Debit products.")
                with st.expander(f"Click here to see the specific Direct Debit products recommendation"):
                    st.warning(f"A model will be implemented in the future to recommend specific Direct Debit products")
            
    except Exception as e:
        st.error(f"Error in customer segmentation: {str(e)}")

    

    
    
    
    
    
    
    
    
    

    

    
    
    
    
    
    
    
    
