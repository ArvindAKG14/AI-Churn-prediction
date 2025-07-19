import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Scikit-learn Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Scikit-learn Model Selection & Evaluation
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, ShuffleSplit, KFold
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve, auc, roc_auc_score, roc_curve,
    make_scorer, log_loss, average_precision_score
)
from sklearn import feature_selection, model_selection, metrics

# Scikit-learn Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Gradient Boosting Libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Data Loading
try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# Data Cleaning (Basic Example, adjust as needed for your dataset)

if 'TotalCharges' in data.columns:
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(subset=['TotalCharges'], inplace=True)

# Conversion of categorical variables
if 'SeniorCitizen' in data.columns and data['SeniorCitizen'].dtype == 'int64':
    data['SeniorCitizen'] = data['SeniorCitizen'].astype(str).replace({'0': 'No', '1': 'Yes'})

if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    if col in data.columns:
        data[col] = data[col].replace({'No internet service': 'No'})
if 'MultipleLines' in data.columns:
    data['MultipleLines'] = data['MultipleLines'].replace({'No phone service': 'No'})

# Plotting Functions

def plot_churn_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='viridis')
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Number of Customers')
    plt.show()

def plot_churn_by_gender(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='gender', hue='Churn', palette='viridis')
    plt.title('Churn Distribution Regarding Gender')
    plt.xlabel('Gender')
    plt.ylabel('Number of Customers')
    plt.legend(title='Churn')
    plt.show()

def plot_categorical_distribution(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, hue='Churn', palette='viridis', order=df[column].value_counts().index)
    plt.title(f'{title} Distribution with Churn')
    plt.xlabel(column)
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df, numerical_col, target_col):
    plt.figure(figsize=(14, 6))

    # Distribution plot
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=numerical_col, hue=target_col, kde=True, palette='viridis')
    plt.title(f'Distribution of {numerical_col}')
    plt.xlabel(numerical_col)
    plt.ylabel('Count')

    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=target_col, y=numerical_col, palette='viridis')
    plt.title(f'{numerical_col} by {target_col}')
    plt.xlabel(target_col)
    plt.ylabel(numerical_col)

    plt.tight_layout()
    plt.show()

# Generate Charts

if 'Churn' in data.columns:
    print("\n--- Generating Churn Distribution Charts ---")
    plot_churn_distribution(data)
else:
    print("\nWarning: 'Churn' column not found. Skipping churn distribution charts.")

if 'gender' in data.columns and 'Churn' in data.columns:
    print("\n--- Generating Churn Distribution Regarding Gender Chart ---")
    plot_churn_by_gender(data)
else:
    print("\nWarning: 'gender' or 'Churn' column not found. Skipping churn by gender chart.")

# Categorical distributions with Churn
categorical_cols_to_plot = {
    'Contract': 'Customer Contract',
    'PaymentMethod': 'Payment Methods',
    'InternetService': 'Internet Services',
    'Dependents': 'Dependent',
    'OnlineSecurity': 'Online Security',
    'SeniorCitizen': 'Senior Citizen',
    'PaperlessBilling': 'Paperless Billing',
    'TechSupport': 'Tech Support'
}

print("\n--- Generating Categorical Distribution Charts with Churn ---")
for col, title in categorical_cols_to_plot.items():
    if col in data.columns and 'Churn' in data.columns:
        plot_categorical_distribution(data, col, title)
    else:
        print(f"Warning: '{col}' or 'Churn' column not found. Skipping {title} chart.")

# Distribution with respect to Charges and Tenure
print("\n--- Generating Distribution Charts with Respect to Charges and Tenure ---")
if 'MonthlyCharges' in data.columns and 'Churn' in data.columns:
    plot_numerical_distributions(data, 'MonthlyCharges', 'Churn')
else:
    print("\nWarning: 'MonthlyCharges' or 'Churn' column not found. Skipping Monthly Charges distribution.")

if 'TotalCharges' in data.columns and 'Churn' in data.columns:
    plot_numerical_distributions(data, 'TotalCharges', 'Churn')
else:
    print("\nWarning: 'TotalCharges' or 'Churn' column not found. Skipping Total Charges distribution.")

if 'tenure' in data.columns and 'Churn' in data.columns:
    plot_numerical_distributions(data, 'tenure', 'Churn')
else:
    print("\nWarning: 'tenure' or 'Churn' column not found. Skipping Tenure distribution.")


# Data Preprocessing
print("\n--- Starting Data Preprocessing ---")

# Separate features (X) and target (y)
if 'Churn' not in data.columns:
    print("Error: 'Churn' column not found in the dataset. Cannot proceed with model training.")
    exit()

X = data.drop('Churn', axis=1)
y = data['Churn']

# Label encode the target variable 'Churn' (Yes/No to 1/0)
le = LabelEncoder()
y = le.fit_transform(y)
print(f"Churn classes encoded: {le.classes_} -> {le.transform(le.classes_)}")

# Identify categorical and numerical columns for preprocessing
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Data preprocessing complete. Data split into training and testing sets.")
print(f"Shape of processed training data: {X_train_processed.shape}")
print(f"Shape of processed test data: {X_test_processed.shape}")


# Model Training and Evaluation 
print("\n--- Starting Model Training and Evaluation ---")

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains a model, makes predictions, and prints its accuracy."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.4f}")
    return accuracy


models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Support Vector Machine": SVC(random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    train_and_evaluate_model(model, X_train_processed, y_train, X_test_processed, y_test, name)
