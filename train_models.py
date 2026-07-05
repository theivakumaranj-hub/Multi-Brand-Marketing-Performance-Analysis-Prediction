import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cleaned_campaign_data.csv')

# Print the exact mathematical distribution
print("=== Target Class Distribution ===")
print(df['Profit_Flag'].value_counts())
print("\n=== Percentage Distribution ===")
print(df['Profit_Flag'].value_counts(normalize=True) * 100)

# Visualize the distribution
# FIXED SEABORN WARNING: Assigned hue and turned off legend
sns.countplot(x='Profit_Flag', data=df, hue='Profit_Flag', palette='Set2', legend=False)
plt.title('Campaign Profit vs Loss Distribution')
plt.show()

# 1. Load the cleaned data
df = pd.read_csv('cleaned_campaign_data.csv')

# 2. Prepare the Features (X)
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

# 5. FEATURE SCALING (Fits on training data, transforms both train and test)
scaler = StandardScaler()                                             # 2. INITIALIZED SCALER
X_train_scaled = scaler.fit_transform(X_train)                        # 3. FIT AND TRANSFORM TRAIN
X_test_scaled = scaler.transform(X_test)                              # 4. TRANSFORM TEST

print("--- TRAINING REGRESSION MODEL (REVENUE) ---")
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_rev_train)                            # 5. USED SCALED DATA
rev_predictions = reg_model.predict(X_test_scaled)

# Calculate Regression Metrics
rmse = np.sqrt(mean_squared_error(y_rev_test, rev_predictions))
mae = mean_absolute_error(y_rev_test, rev_predictions)
r2 = r2_score(y_rev_test, rev_predictions)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared (R2): {r2:.4f}\n")

print("--- TRAINING CLASSIFICATION MODEL (PROFIT/LOSS) ---")
# Using Logistic Regression for binary classification (1 = Profit, 0 = Loss)
# Note: You can lower max_iter or leave it; with scaling, it will converge instantly!
clf_model = LogisticRegression(class_weight='balanced', random_state=42) # 6. ADDED BALANCING
clf_model.fit(X_train_scaled, y_prof_train)                           # 7. USED SCALED DATA
prof_predictions = clf_model.predict(X_test_scaled)

# Calculate Classification Metrics
accuracy = accuracy_score(y_prof_test, prof_predictions)
precision = precision_score(y_prof_test, prof_predictions)
recall = recall_score(y_prof_test, prof_predictions)
f1 = f1_score(y_prof_test, prof_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")

# 6. Save the models, the scaler, and the final feature list for the Streamlit App
print("Saving models for Streamlit deployment...")
joblib.dump(reg_model, 'revenue_regression_model.pkl')
joblib.dump(clf_model, 'profit_classification_model.pkl')
joblib.dump(scaler, 'scaler.pkl') 
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print("Success! Models trained, metrics calculated, and files saved.")
print(df['Profit_Flag'].value_counts())
