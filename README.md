# 💼 Employee Salary Prediction

A machine learning project to predict whether an employee earns **>50K or <=50K** per year, using demographic and work-related features.  
Built with **Logistic Regression** and deployed via **Streamlit**.

---

## 📊 Dataset

- Source: UCI Adult Dataset  
- File used: `adult 3.csv`  
- Contains features like age, education, workclass, marital status, occupation, etc.

---

## ⚙️ Technologies Used

- Python  
- pandas, numpy  
- scikit-learn  
- joblib  
- Streamlit

---

## 🧠 Model

- **Preprocessing:**
  - Handled missing values (`?`)
  - Encoded categorical variables using OneHotEncoder
  - Scaled numerical features

- **Model Used:** Logistic Regression  
- **Target Variable:** `income` (`>50K` or `<=50K`)

- **Saved Files:**
  - `salary_prediction_pipeline.pkl` – Trained ML pipeline  
  - `salary_label_encoder.pkl` – Label encoder for target

---

## 🚀 Streamlit App

The model is deployed as an interactive web app using **Streamlit**.

---
