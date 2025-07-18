
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- 1. Load the dataset ---
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("adult 3.csv")

    # Strip column names and values
    df.columns = df.columns.str.strip().str.replace('-', '_')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Replace '?' with mode
    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Split into features and target
    X = df.drop('income', axis=1)
    y = df['income']

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Preprocessing
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train the model
    pipeline.fit(X, y_encoded)
    return pipeline, label_encoder, X.columns

pipeline, label_encoder, model_columns = load_and_train_model()

# --- 2. Streamlit UI ---
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")
st.title("💰 Employee Salary Prediction")
st.markdown("Use the input fields below to predict if an employee's salary is >50K or <=50K.")

# Define options
WORKCLASS_OPTIONS = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Without-pay', 'Never-worked']
EDUCATION_OPTIONS = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th', 'Doctorate']
MARITAL_STATUS_OPTIONS = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
OCCUPATION_OPTIONS = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
RELATIONSHIP_OPTIONS = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
RACE_OPTIONS = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
SEX_OPTIONS = ['Male', 'Female']
NATIVE_COUNTRY_OPTIONS = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Japan', 'China', 'Cuba', 'Jamaica', 'Italy', 'South', 'Columbia', 'Vietnam', 'Dominican-Republic', 'Guatemala', 'England', 'Taiwan', 'Haiti', 'Portugal', 'Ecuador', 'France', 'Poland', 'Peru', 'Iran', 'Honduras', 'Greece', 'Nicaragua', 'Scotland', 'Thailand', 'Ireland', 'Trinadad&Tobago', 'Hungary', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hong', 'Cambodia']

# Form UI
with st.form("prediction_form"):
    st.header("Employee Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35)
        sex = st.selectbox("Gender", options=SEX_OPTIONS)
        race = st.selectbox("Race", options=RACE_OPTIONS)
    with col2:
        workclass = st.selectbox("Work Class", options=WORKCLASS_OPTIONS)
        occupation = st.selectbox("Occupation", options=OCCUPATION_OPTIONS)
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
        native_country = st.selectbox("Native Country", options=NATIVE_COUNTRY_OPTIONS)
    with col3:
        education = st.selectbox("Education", options=EDUCATION_OPTIONS)
        education_num = st.number_input("Educational Number", min_value=1, max_value=16, value=10)
        marital_status = st.selectbox("Marital Status", options=MARITAL_STATUS_OPTIONS)
        relationship = st.selectbox("Relationship", options=RELATIONSHIP_OPTIONS)

    st.subheader("Financial Details")
    col_fin1, col_fin2, col_fin3 = st.columns(3)
    with col_fin1:
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    with col_fin2:
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    with col_fin3:
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1000, value=200000)

    submit_button = st.form_submit_button(label="Predict Salary")

if submit_button:
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

    input_df = pd.DataFrame([input_data])
    prediction_encoded = pipeline.predict(input_df)[0]
    predicted_salary = label_encoder.inverse_transform([prediction_encoded])[0]

    st.subheader("Prediction Result:")
    if predicted_salary.strip() == '>50K':
        st.success(f"The model predicts the salary is *{predicted_salary}*")
    else:
        st.info(f"The model predicts the salary is *{predicted_salary}*")
