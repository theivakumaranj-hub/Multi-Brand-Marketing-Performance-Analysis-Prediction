import pandas as pd

# --- STEP 1: DATA COLLECTION ---
nykaa_df = pd.read_csv('nykaa_campaign_data_with_nulls.csv')
purplle_df = pd.read_csv('purplle_campaign_data_with_nulls.csv')
tira_df = pd.read_csv('tira_campaign_data_with_nulls.csv')

df = pd.concat([nykaa_df, purplle_df, tira_df], ignore_index=True)
print(f"Original Row Count: {len(df)}")

# --- STEP 2: DATA PREPROCESSING ---
# 1. Remove duplicate records
df = df.drop_duplicates()

# 2. Handle missing values by dropping them
df = df.dropna()
print(f"Row Count after dropping nulls and duplicates: {len(df)}")

# 3. Standardize the Date format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# --- STEP 3: FEATURE ENGINEERING ---

# 1. Create Profit/Loss flag based on the ORIGINAL data distribution first
# (This ensures the ML model gets a healthy mix of both profits and losses to learn from)
df['Profit_Flag'] = df['ROI'].apply(lambda x: 1 if x > 0 else 0)

# 2. Validate and Recalculate ROI mathematically
df['ROI'] = ((df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost']) * 100

# 3. Multi-label encoding for Channel_Used
df['Channel_Used'] = df['Channel_Used'].str.replace(', ', ',') # Clean up spacing
channel_dummies = df['Channel_Used'].str.get_dummies(sep=',')
df = pd.concat([df, channel_dummies], axis=1)

# Drop the original 'Channel_Used' column
df = df.drop('Channel_Used', axis=1)

# --- SAVE THE PREPARED DATA ---
df.to_csv('cleaned_campaign_data.csv', index=False)
print("\nDataset successfully cleaned, engineered, and saved as 'cleaned_campaign_data.csv'!")

# Check the new columns to verify its worked
print("\n--- FINAL COLUMNS ---")
print(df.columns.tolist())
