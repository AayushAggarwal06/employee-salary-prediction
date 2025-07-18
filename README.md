# 💼 Employee Salary Prediction

A machine learning project that predicts whether an employee earns **>50K or <=50K** per year based on demographic and work-related features.  
This project uses **Logistic Regression** and is deployed locally through a **Streamlit web app**.

---

## 📊 Dataset

- Source: UCI Adult Dataset  
- File used: `adult 3.csv`  
- Contains features such as age, education, workclass, marital status, occupation, etc.

---

## ⚙️ Technologies Used

- Python  
- pandas, numpy  
- scikit-learn  
- joblib  
- streamlit

---

## 🧠 Model Workflow

1. **Data Cleaning:**  
   - Removed extra spaces and missing values represented by `'?'`  
   - Handled categorical encoding and numerical scaling  
2. **Model Building:**  
   - Trained using Logistic Regression inside a Scikit-learn pipeline  
   - Evaluated using classification metrics  
3. **Deployment:**  
   - Model training is embedded in the Streamlit app
   - Live deployed with Streamlit Cloud

---

## 🚀 How to Use the Live App

👉 **Live App Link:**  
[https://employee-salary-prediction-ai.streamlit.app/](https://employee-salary-prediction-ai.streamlit.app/)

You can enter employee details and get a real-time salary category prediction (`<=50K` or `>50K`).

---

## 📚 References

- [UCI Machine Learning Repository – Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
- [scikit-learn documentation](https://scikit-learn.org/stable/)  
- [Streamlit documentation](https://docs.streamlit.io/)

---

## 🙏 Acknowledgement

This project was developed as part of the **Edunet Foundation AI/ML Internship 2025**, under the guidance of **GJUST – Department of CSE**.

---
