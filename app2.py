import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the trained model and label encoder ---
MODEL_PATH = 'salary_prediction_pipeline.pkl'
ENCODER_PATH = 'salary_label_encoder.pkl'

# Change @st.cache_resource to @st.cache_data
@st.cache_data 
def load_model():
    """Load the trained machine learning pipeline and label encoder."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        return pipeline, label_encoder
    except FileNotFoundError:
        st.error(f"Model files not found. Please ensure '{MODEL_PATH}' and '{ENCODER_PATH}' are in the correct location.")
        return None, None

pipeline, label_encoder = load_model()

# --- 2. Define standard features and options (typical from the Adult dataset) ---

# Define typical options for categorical features for the UI dropdowns
WORKCLASS_OPTIONS = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Without-pay', 'Never-worked']
EDUCATION_OPTIONS = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th', 'Doctorate']
MARITAL_STATUS_OPTIONS = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
OCCUPATION_OPTIONS = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
RELATIONSHIP_OPTIONS = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
RACE_OPTIONS = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
SEX_OPTIONS = ['Male', 'Female']
NATIVE_COUNTRY_OPTIONS = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Japan', 'China', 'Cuba', 'Jamaica', 'Italy', 'South', 'Columbia', 'Vietnam', 'Dominican-Republic', 'Guatemala', 'England', 'Taiwan', 'Haiti', 'Portugal', 'Ecuador', 'France', 'Poland', 'Peru', 'Iran', 'Honduras', 'Greece', 'Nicaragua', 'Scotland', 'Thailand', 'Ireland', 'Trinadad&Tobago', 'Hungary', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hong', 'Cambodia']

# --- 3. Streamlit UI Layout ---

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

st.title("💰 Employee Salary Prediction")
st.markdown("Use the input fields below to predict if an employee's salary is >50K or <=50K.")

if pipeline is None or label_encoder is None:
    st.warning("Model cannot be loaded. Please check the files and paths.")
else:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.header("Employee Details")

        # Layout columns for inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
            sex = st.selectbox("gender", options=SEX_OPTIONS)
            race = st.selectbox("Race", options=RACE_OPTIONS)
            
        with col2:
            st.subheader("Work Information")
            workclass = st.selectbox("Work Class", options=WORKCLASS_OPTIONS)
            occupation = st.selectbox("Occupation", options=OCCUPATION_OPTIONS)
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
            native_country = st.selectbox("Native Country", options=NATIVE_COUNTRY_OPTIONS)
            
        with col3:
            st.subheader("Education & Status")
            education = st.selectbox("Education", options=EDUCATION_OPTIONS)
            education_num = st.number_input("Educational Number", min_value=1, max_value=16, value=10)
            marital_status = st.selectbox("Marital Status", options=MARITAL_STATUS_OPTIONS)
            relationship = st.selectbox("Relationship", options=RELATIONSHIP_OPTIONS)
        
        # Additional features in a separate section
        st.subheader("Financial Details")
        col_fin1, col_fin2, col_fin3 = st.columns(3)
        
        with col_fin1:
            capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        with col_fin2:
            capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        with col_fin3:
            fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1000, value=200000)

        # Submit button
        submit_button = st.form_submit_button(label="Predict Salary")

    # --- 4. Prediction Logic ---

    if submit_button:
        # Create a dictionary from the user inputs
        input_data = {
            'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'educational_num': education_num,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'gender': sex,
            'capital_gain': capital_gain,
            'capital_loss': capital_loss,
            'hours_per_week': hours_per_week,
            'native_country': native_country
        }

        # Convert dictionary to DataFrame for the pipeline
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        try:
            prediction_encoded = pipeline.predict(input_df)[0]
            predicted_salary = label_encoder.inverse_transform([prediction_encoded])[0]

            # Display result
            st.subheader("Prediction Result:")
            
            if predicted_salary.strip() == '>50K':
                st.success(f"The model predicts the salary is *{predicted_salary}*")
            else:
                st.info(f"The model predicts the salary is *{predicted_salary}*")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
