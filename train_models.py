import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Load the cleaned data
df = pd.read_csv('cleaned_campaign_data.csv')

# 2. Prepare the Features (X)
# We drop Campaign_ID and Date as they aren't predictive.
# We explicitly drop ROI to prevent data leakage per project guidelines.
X = df.drop(columns=['Campaign_ID', 'Date', 'ROI', 'Revenue', 'Profit_Flag'])

# Convert categorical text columns into numbers (One-Hot Encoding)
categorical_cols = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 3. Prepare the Targets (y)
y_revenue = df['Revenue']        # For Regression
y_profit = df['Profit_Flag']     # For Classification

# 4. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_rev_train, y_rev_test, y_prof_train, y_prof_test = train_test_split(
    X, y_revenue, y_profit, test_size=0.2, random_state=42
)

print("--- TRAINING REGRESSION MODEL (REVENUE) ---")
reg_model = LinearRegression()
reg_model.fit(X_train, y_rev_train)
rev_predictions = reg_model.predict(X_test)

# Calculate Regression Metrics
rmse = np.sqrt(mean_squared_error(y_rev_test, rev_predictions))
mae = mean_absolute_error(y_rev_test, rev_predictions)
r2 = r2_score(y_rev_test, rev_predictions)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared (R2): {r2:.4f}\n")

print("--- TRAINING CLASSIFICATION MODEL (PROFIT/LOSS) ---")
# Using Logistic Regression for binary classification (1 = Profit, 0 = Loss)
clf_model = LogisticRegression(max_iter=10000)
clf_model.fit(X_train, y_prof_train)
prof_predictions = clf_model.predict(X_test)

# Calculate Classification Metrics
accuracy = accuracy_score(y_prof_test, prof_predictions)
precision = precision_score(y_prof_test, prof_predictions)
recall = recall_score(y_prof_test, prof_predictions)
f1 = f1_score(y_prof_test, prof_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")

# 5. Save the models and the final feature list for the Streamlit App
print("Saving models for Streamlit deployment...")
joblib.dump(reg_model, 'revenue_regression_model.pkl')
joblib.dump(clf_model, 'profit_classification_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print("Success! Models trained, metrics calculated, and files saved.")