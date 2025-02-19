#!/usr/bin/env python
# auto_ml_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np

# Import ML models and metrics.
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Import Hugging Face transformers.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Set page config; using the sidebar for navigation.
st.set_page_config(page_title="Auto ML Pipeline with LLM Assistance", layout="wide")

# Sidebar navigation between pages.
page = st.sidebar.radio("Select Page", options=["ML Pipeline", "Interpretation"])

# ------------------------------------------------------------------------------
# Load primary NLP model for code suggestions (using flan-t5-small).
@st.cache_resource(show_spinner=False)
def load_primary_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

primary_llm = load_primary_llm()

# Load an alternate NLP model for interpretation (using t5-base).
@st.cache_resource(show_spinner=False)
def load_interpreter_llm():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

interpreter_llm = load_interpreter_llm()

# Functions to get responses.
def get_llm_response(prompt, max_length=200):
    response = primary_llm(prompt, max_length=max_length, do_sample=False)
    return response[0]['generated_text']

def get_interpretation_response(prompt, max_length=512):
    response = interpreter_llm(prompt, max_length=max_length, do_sample=False)
    return response[0]['generated_text']

# ------------------------------------------------------------------------------
# Data Cleaning Function.
def clean_data(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    num_cols = df_clean.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean

# ------------------------------------------------------------------------------
# Function to run the ML model.
def run_ml_model(df, target_col, feature_list, model_choice):
    if not feature_list:
        st.error("No features with sufficient contribution were found!")
        return None

    X = df[feature_list]
    y = df[target_col]

    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx, :], X.iloc[split_idx:, :]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results = {
            'algorithm': model_choice,
            'features': feature_list,
            'coefficients': dict(zip(feature_list, model.coef_)) if hasattr(model, 'coef_') else None,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else None,
            'rmse': rmse,
            'r2_score': r2,
            'predictions': y_pred
        }
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accu = accuracy_score(y_test, y_pred)
        results = {
            'algorithm': model_choice,
            'features': feature_list,
            'coefficients': dict(zip(feature_list, model.coef_[0])) if hasattr(model, 'coef_') else None,
            'intercept': model.intercept_[0] if hasattr(model, 'intercept_') else None,
            'accuracy': accu,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': y_pred
        }
    elif model_choice == "Random Forest Classifier":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accu = accuracy_score(y_test, y_pred)
        results = {
            'algorithm': model_choice,
            'features': feature_list,
            'accuracy': accu,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': y_pred
        }
    else:
        st.error("Unsupported ML algorithm selected!")
        return None
    return results

# ------------------------------------------------------------------------------
# Page: ML Pipeline
if page == "ML Pipeline":
    st.title("Auto ML Pipeline with LLM Assistance")
    st.header("1. Upload Your Numerical Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
            st.write("Dataset Summary:")
            st.write(df.describe())
            st.session_state['df_raw'] = df

            st.header("2. Data Cleaning & Preprocessing")
            if st.button("Suggest Data Cleaning Code with LLM"):
                prompt = (
                    "I have a dataset with the following summary:\n" +
                    df.describe().to_string() +
                    "\nSuggest data cleaning steps in Python using pandas. Include removing duplicates and handling missing numerical values using median imputation."
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

            st.header("3. Select Target, Evaluate Feature Contribution & Run ML Model")
            if 'df_clean' in st.session_state:
                df_clean = st.session_state['df_clean']
                numeric_cols = list(df_clean.select_dtypes(include=["number"]).columns)
                target_col = st.selectbox("Select the target column", options=numeric_cols)
                if st.button("Evaluate Feature Contribution"):
                    correlations = df_clean.corr()[target_col].drop(target_col)
                    st.subheader("Correlation of Features with Target")
                    st.write(correlations.sort_values(ascending=False))
                    threshold = 0.2
                    relevant_features = correlations[correlations.abs() > threshold].index.tolist()
                    if not relevant_features:
                        st.error("No features found with an absolute correlation > " + str(threshold))
                    else:
                        st.session_state['selected_features'] = relevant_features
                        st.write("Selected features (abs(correlation) > ", threshold, "):", relevant_features)
                        st.bar_chart(correlations[relevant_features])
                st.subheader("Select ML Algorithm")
                if pd.api.types.is_numeric_dtype(df_clean[target_col]):
                    if df_clean[target_col].nunique() < 10:
                        model_options = ["Logistic Regression", "Random Forest Classifier", "Linear Regression"]
                    else:
                        model_options = ["Linear Regression"]
                else:
                    model_options = ["Logistic Regression", "Random Forest Classifier"]
                model_choice = st.radio("Choose the ML algorithm to use", options=model_options)
                if st.button("Suggest ML Model Code with LLM"):
                    sel_feats = st.session_state.get('selected_features', [])
                    prompt_ml = (
                        f"I have a cleaned dataset and plan to predict '{target_col}' using only the features: {sel_feats}. " +
                        f"Provide a Python code snippet that uses {model_choice} from scikit-learn, including splitting the data into training and test sets and calculating evaluation metrics."
                    )
                    ml_suggestion = get_llm_response(prompt_ml, max_length=350)
                    st.subheader("LLM Suggested ML Model Code:")
                    st.code(ml_suggestion, language='python')
                if st.button("Run ML Model"):
                    sel_feats = st.session_state.get('selected_features', [])
                    results = run_ml_model(df_clean, target_col, sel_feats, model_choice)
                    if results:
                        st.subheader("ML Model Results")
                        st.write("Algorithm Selected:", results.get('algorithm', ''))
                        st.write("Using features:", results.get('features', []))
                        if model_choice == "Linear Regression":
                            st.write("Feature Coefficients (including intercept):")
                            st.write({"intercept": results['intercept'], **results['coefficients']})
                            st.write(f"RMSE: {results['rmse']:.4f}")
                            st.write(f"R2 Score: {results['r2_score']:.4f}")
                        else:
                            st.write("Accuracy:", results.get('accuracy', ''))
                            st.write("Classification Report:", results.get('classification_report', ''))
                        # Build a verbose interpretation prompt.
                        if model_choice == "Linear Regression":
                            result_text = (
                                "Please provide a very detailed and verbose interpretation of the following model results. "
                                "Explain what the intercept and each coefficient indicate about the relationship between the features and the target variable. "
                                "Also describe what the RMSE and R2 score tell you about the overall performance of the model, including potential limitations. "
                                f"Model details: The Linear Regression model was trained to predict '{target_col}' using features: {results['features']}. "
                                f"It has an intercept of {results['intercept']:.4f} and coefficients: {results['coefficients']}. "
                                f"The model achieved an RMSE of {results['rmse']:.4f} and an R2 score of {results['r2_score']:.4f}."
                            )
                        else:
                            result_text = (
                                "Please provide a very detailed and verbose interpretation of the following classification model results. "
                                "Explain what the accuracy and the details within the classification report indicate about the model's performance. "
                                "Discuss how each feature might influence the predicted outcome and mention any potential issues with the model. "
                                f"Model details: The {model_choice} model was trained to predict '{target_col}' using features: {results['features']}. "
                                f"It achieved an accuracy of {results.get('accuracy', 0):.4f}."
                            )
                        st.session_state['interpretation_prompt'] = result_text
                        st.success("ML Model run complete! To view the verbose interpretation, select the 'Interpretation' page from the sidebar.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Awaiting CSV file upload...")
# ------------------------------------------------------------------------------
# Page: Interpretation
elif page == "Interpretation":
    st.title("ML Model Interpretation")
    if 'interpretation_prompt' in st.session_state:
        interpretation_prompt = st.session_state['interpretation_prompt']
        st.write("Interpretation prompt used:")
        st.code(interpretation_prompt)
        if st.button("Generate Verbose Interpretation", key="interpret_alt"):
            interpretation = get_interpretation_response("Provide a detailed interpretation of the following ML model results:\n" + interpretation_prompt, max_length=512)
            st.session_state['interpretation'] = interpretation
            st.subheader("Verbose LLM Interpretation of Model Results:")
            st.write(interpretation)
        elif 'interpretation' in st.session_state:
            st.subheader("Verbose LLM Interpretation of Model Results:")
            st.write(st.session_state['interpretation'])
        else:
            st.info("Click the button above to generate the verbose interpretation.")
    else:
        st.info("No interpretation available yet. Please run the ML Pipeline and generate results first.")