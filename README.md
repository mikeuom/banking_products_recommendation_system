# Retail Banking Products Recommendation Pipeline

This repository contains a comprehensive banking products recommendation system with two main components: model development and user interface.

## Project Structure

### 1. Model Development (`notebooks_models_development/`)
This directory contains all the notebooks and code used for developing and training the recommendation models.

```
notebooks_models_development/
├── banking_products_category_recommendation_model/  # Model for general product category recommendations
│   ├── banking_products_category_model.ipynb       # Main model development notebook
│   ├── best_multilabel_model.pkl                   # Trained model file
│   └── customer_data_recommendations.csv           # Processed customer data
│
├── customer_segmentation_model/                    # Customer segmentation analysis and modeling
│   ├── customer_segmentation_model.ipynb           # Segmentation model development
│   ├── kmeans_customer_segments.pkl                # Trained segmentation model
│   └── transaction_data_recommendations.csv        # Transaction data for recommendations
│
├── credit_card_products_prediction_model/          # Credit card product recommendations
│   ├── CreditCardSubtypeRecommendation_Unsupervised.ipynb        # Main credit card model
│   ├── CreditCardSubtypeRecommendation_Unsupervised_Documentation.ipynb  # Model documentation
│   ├── kmeans_model_creditcardsubtypeunsupervised.pkl            # Trained credit card model
│   └── CreditCardSpendingCategories.txt            # Credit card spending categories
│
├── loans_products_prediction_model/                # Loan product recommendations
│   ├── loan model.ipynb                            # Loan recommendation model
│   ├── loan_recommendations_analysis.png           # Analysis visualization
│   └── loan_recommendations.csv                    # Loan recommendations data
│
├── dataset/                                        # Raw and processed datasets
│
├── feature_engineering/                           # Feature engineering scripts and notebooks
│   ├── feature engineer.ipynb                      # Feature engineering notebook
│   └── clean_customer_dataNEW.csv                 # Cleaned customer data
│
├── eda/                                           # Exploratory Data Analysis notebooks
│   └── EDA.ipynb                                  # Main EDA notebook
│
└── pipeline_design.jpg                           # Visual representation of the pipeline architecture
```

### 2. User Interface (`user-interface-streamlit/`)
This directory contains the Streamlit-based web application for user interaction with the recommendation system.

```
user-interface-streamlit/
├── app/                                           # Main application code
│   ├── main.py                                    # Main Streamlit application
│   ├── assets/                                    # Static assets (images, etc.)
│   │   ├── vertical_pipeline.jpg                  # Pipeline visualization
│   │   └── pca_4d_interactive.html               # Interactive PCA visualization
│   │
│   └── utils/                                     # Utility functions
│       ├── product_category_recommender.py        # Product category recommendation logic
│       ├── loan_product_recommender.py            # Loan product recommendation logic
│   │   └── feature_engineering.py                 # Feature engineering utilities
│   │   └── credit_card_recommender.py             # Credit card recommendation logic
│   │   └── customer_segmenter.py                  # Customer segmentation logic
│   │
│   ├── data/                                          # Data files used by the application
│   │   ├── demo_data/                                 # Sample data for demonstration
│   │   └── background_data/                          # Background data for recommendations
│   │
│   ├── models/                                        # Trained model files
│   │   ├── primary_model/                            # Banking Products Category recommendation model
│   │   ├── loan_model/                               # Loan recommendation model
│   │   ├── customer_segment_model/                   # Customer segmentation model
│   │   └── credit_card_model/                        # Credit card recommendation model
│   │
│   └── requirements.txt                               # Python dependencies
│
└── venv/                                          # Virtual environment (not tracked in git)
```

## Getting Started

1. First and foremost, dowload the required CSV files in section Data Files

2. For model development:
   - Navigate to `notebooks_models_development/`
   - Follow the notebooks in order: EDA → Feature Engineering → Model Development

3. For the user interface:
   - Navigate to `user-interface-streamlit/`
   - Install dependencies: `pip install -r requirements.txt`
   - Run the Streamlit app: `streamlit run app/main.py`

## Data Files
The CSV files used in this project are not uploaded to GitHub due to their size and privacy considerations. These files are available via Google Drive in link mentioned below. Users need to:

Link: https://drive.google.com/drive/folders/1cNBnl8jKsHeZNzCEufggGPUPYkDGfUsP?usp=sharing

1. Download the CSV files from the provided Google Drive link
2. Place them in their corresponding folders as shown in the project structure

The project will not function properly without these data files in place, especially the user interface

## Note
- The `venv/` directory is not tracked in git and should be created locally
- Make sure to keep the model files in sync between the development and UI components 
