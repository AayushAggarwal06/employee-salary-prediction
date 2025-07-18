import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
try:
    df = pd.read_csv('adult 3.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'adult 3.csv' was not found.")
    exit()

# Display the first few rows and column information
print("\n--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())

# 1.1 Data Cleaning: Handle spaces in column names and string data
# Many Adult datasets have spaces in column names and leading/trailing spaces in string values.

df.columns = df.columns.str.strip().str.replace('-', '_')

# Remove leading/trailing spaces from all object columns
for column in df.select_dtypes(include='object').columns:
    df[column] = df[column].str.strip()

# 1.2 Identify and handle missing values
# Missing values are often represented as '?' in this dataset.
print("\n--- Missing Value Check (Count of '?') ---")
missing_values_count = (df == '?').sum()
print(missing_values_count[missing_values_count > 0])

# Replace '?' with NaN and then impute or drop. We will impute with the mode for categorical features.
df.replace('?', np.nan, inplace=True)

# Impute missing values (for simplicity, we'll use the mode for categorical columns)
for column in df.select_dtypes(include='object').columns:
    if df[column].isnull().sum() > 0:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

# 1.3 Feature Engineering and Preprocessing

# Define features (X) and target (y)
# The target variable is 'income'
X = df.drop('income', axis=1)
y = df['income']

# Encode the target variable (<=50K and >50K)
# ' <=50K' will be 0, ' >50K' will be 1
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nTarget Classes:", le.classes_)

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# 1.4 Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("\nTraining and testing data split:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 2.1 Define preprocessing steps

# Numerical transformer: Standard scaling
numerical_transformer = StandardScaler()

# Categorical transformer: One-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns if they exist (although none expected here)
)

# 2.2 Create the ML Pipeline
# We'll use Logistic Regression for classification
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 2.3 Train the model
print("\n--- Training the model ---")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# 3.1 Make predictions on the test set
y_pred = pipeline.predict(X_test)

# 3.2 Evaluate the model
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 4.1 Save the trained pipeline and LabelEncoder
model_filename = 'salary_prediction_pipeline.pkl'
label_encoder_filename = 'salary_label_encoder.pkl'

joblib.dump(pipeline, model_filename)
joblib.dump(le, label_encoder_filename)

print(f"\nModel saved as '{model_filename}'")
print(f"Label Encoder saved as '{label_encoder_filename}'")

# 4.2 Define a function to load the model and make a prediction
def predict_salary(data):
    """
    Loads the trained model and makes a salary prediction based on input data.

    Args:
        data (dict): Dictionary containing the features of an employee.
                     Keys should match the column names in the training data.

    Returns:
        str: Predicted income category ('>50K' or '<=50K').
    """
    try:
        # Load the model and label encoder
        model = joblib.load(model_filename)
        label_encoder = joblib.load(label_encoder_filename)

        # Convert input dictionary to a pandas DataFrame (required for the pipeline)
        # We need to ensure the columns are in the correct order, which the DataFrame conversion handles if the dictionary keys match.
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction_encoded = model.predict(input_df)[0]

        # Decode the prediction
        predicted_income = label_encoder.inverse_transform([prediction_encoded])[0]

        return predicted_income

    except Exception as e:
        return f"An error occurred during prediction: {e}"

# 4.3 Example usage of the prediction function
print("\n--- Example Prediction ---")

# Example features based on the dataset columns
# Ensure all columns used during training are present
example_employee = {
    'age': 39,
    'workclass': 'State-gov',
    'fnlwgt': 77516,
    'education': 'Bachelors',
    'educational_num': 13,
    'marital_status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'gender': 'Male',
    'capital_gain': 2174,
    'capital_loss': 0,
    'hours_per_week': 40,
    'native_country': 'United-States'
}

predicted_salary = predict_salary(example_employee)
print(f"The predicted salary for the example employee is: {predicted_salary}")
