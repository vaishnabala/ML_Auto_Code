#!/usr/bin/env python
# auto_ml_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Auto ML Pipeline with LLM Assistance", layout="wide")

# ------------------------------------------------------------------------------
# Load an open–source LLM using Hugging Face transformers.
# Here we use "google/flan-t5-small" for demonstration.
@st.cache_resource(show_spinner=False)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return llm_pipe

llm = load_llm()

# Function to generate LLM response for a given prompt.
def get_llm_response(prompt, max_length=200):
    # You may experiment with do_sample=True for diversity.
    response = llm(prompt, max_length=max_length, do_sample=False)
    return response[0]['generated_text']

# ------------------------------------------------------------------------------
# Data cleaning function: For demonstration we drop duplicates,
# fill missing values (with median for numerical columns) and reset index.
def clean_data(df):
    df_clean = df.copy()
    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    # For numeric columns, fill missing with median
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    # Optionally, drop any remaining missing values
    df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean

# ------------------------------------------------------------------------------
# Function to apply a simple ML model (Linear Regression).
def run_ml_model(df, target_col):
    # Prepare data: Use all numeric columns except the target as features.
    feature_cols = [col for col in df.columns if col != target_col and pd.api.types.is_numeric_dtype(df[col])]
    if len(feature_cols) == 0:
        st.error("No suitable feature columns found!")
        return None
    X = df[feature_cols]
    y = df[target_col]
    
    # Split the data into training and testing parts (here, use 80/20 split)
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx, :], X.iloc[split_idx:, :]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Save results (coefficients and scores)
    results = {
        'features': feature_cols,
        'coefficients': dict(zip(feature_cols, model.coef_)),
        'intercept': model.intercept_,
        'rmse': rmse,
        'r2_score': r2
    }
    
    return results

# ------------------------------------------------------------------------------
# Streamlit User Interface
st.title("Auto ML Pipeline with LLM Assistance")

# Section 1: Data Upload
st.header("1. Upload Your Numerical Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write("Dataset Summary:")
        st.write(df.describe())
        
        # Save the dataframe in session state for later steps.
        st.session_state['df_raw'] = df
        
        # ------------------------------------------------------------------------------
        # Section 2: Data Cleaning Recommendation & Execution
        st.header("2. Data Cleaning & Preprocessing")
        
        if st.button("Suggest Data Cleaning Code with LLM"):
            # Create a prompt for the LLM based on the dataset info.
            prompt = (
                "I have a dataset with the following summary: \n"
                f"{df.describe().to_string()}\n"
                "Suggest appropriate data cleaning and preprocessing steps in Python using pandas. "
                "Please include code for dropping duplicates, handling missing numerical values (with median imputation), "
                "and any other improvements."
            )
            suggestion = get_llm_response(prompt, max_length=300)
            st.subheader("LLM Suggested Data Cleaning Code:")
            st.code(suggestion, language='python')
        
        if st.button("Apply Data Cleaning"):
            df_clean = clean_data(df)
            st.session_state['df_clean'] = df_clean
            st.success("Data cleaning applied successfully!")
            st.write("Cleaned Data Preview:")
            st.dataframe(df_clean.head())
       
        # ------------------------------------------------------------------------------
        # Section 3: ML Model Suggestion & Execution
        st.header("3. Choose Target Column & Run Machine Learning Model")
        if 'df_clean' in st.session_state:
            df_clean = st.session_state['df_clean']
            
            # Let the user select the target column from numeric columns.
            numeric_cols = list(df_clean.select_dtypes(include=['number']).columns)
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns (one target and at least one feature) to run the model.")
            else:
                target_col = st.selectbox("Select the target column for ML model", options=numeric_cols)
                
                if st.button("Suggest ML Model Code with LLM"):
                    prompt_ml = (
                        f"I have a cleaned dataset with numerical columns: {numeric_cols}. "
                        f"I plan to predict the target column '{target_col}' using the other numerical features. "
                        "Suggest a Python code snippet that uses scikit-learn to perform a regression using LinearRegression. "
                        "The code should split the data into training and testing sets, train the model, and compute evaluation metrics (RMSE and R2 score)."
                    )
                    ml_suggestion = get_llm_response(prompt_ml, max_length=350)
                    st.subheader("LLM Suggested ML Model Code:")
                    st.code(ml_suggestion, language='python')
                
                if st.button("Run ML Model"):
                    results = run_ml_model(df_clean, target_col)
                    if results is not None:
                        st.subheader("ML Model Results")
                        st.write("Feature Coefficients (including intercept):")
                        st.write({"intercept": results['intercept'], **results['coefficients']})
                        st.write(f"RMSE: {results['rmse']:.4f}")
                        st.write(f"R2 Score: {results['r2_score']:.4f}")
                        
                        # ------------------------------------------------------------------------------
                        # Section 4: LLM Interpretation of ML Model Results
                        st.header("4. LLM Interpretation of the Model Results")
                        if st.button("Interpret ML Results with LLM"):
                            result_text = (
                                f"The linear regression model was trained to predict '{target_col}' using the features: {results['features']}. "
                                f"The obtained coefficients (including an intercept of {results['intercept']:.4f}) yield an RMSE of {results['rmse']:.4f} "
                                f"and an R2 score of {results['r2_score']:.4f}. "
                                "Please provide an interpretation of these results."
                            )
                            interpretation_prompt = (
                                "Interpret the following ML model results: \n" + result_text +
                                "\nExplain what the performance metrics indicate and what the coefficients may imply about feature importance."
                            )
                            interpretation = get_llm_response(interpretation_prompt, max_length=300)
                            st.subheader("LLM Interpretation of Model Results:")
                            st.write(interpretation)
        else:
            st.info("Please apply data cleaning first to proceed with ML modelling.")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting CSV file upload...")