import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned data
df = pd.read_csv('cleaned_campaign_data.csv')

# Extract Brand Name from Campaign_ID (Assuming format like NY-CMP-1000)
df['Brand'] = df['Campaign_ID'].str.split('-').str[0].map({'NY': 'Nykaa', 'PU': 'Purplle', 'TI': 'Tira'})
df['Brand'] = df['Brand'].fillna('Other')

print("Generating EDA Visualizations...")

# 2. Correlation Heatmap: Spend, Clicks, Revenue, and ROI
plt.figure(figsize=(8, 6))
correlation_cols = ['Acquisition_Cost', 'Clicks', 'Impressions', 'Revenue', 'ROI']
sns.heatmap(df[correlation_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Relationship: Spend, Clicks, Revenue, and ROI')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()

# 3. Channel Effectiveness: Total Revenue by Channel
channels = ['Email', 'Facebook', 'Google', 'Instagram', 'WhatsApp', 'YouTube']
channel_revenue = {channel: df[df[channel] == 1]['Revenue'].sum() for channel in channels}

plt.figure(figsize=(10, 5))
sns.barplot(x=list(channel_revenue.keys()), y=list(channel_revenue.values()), palette='viridis')
plt.title('Total Revenue by Marketing Channel')
plt.ylabel('Total Revenue')
plt.tight_layout()
plt.savefig('eda_channel_revenue.png')
plt.close()

# 4. Profit vs Loss Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Profit_Flag', palette='Set2')
plt.title('Campaign Profit (1) vs Loss (0) Distribution')
plt.xticks(ticks=[0, 1], labels=['Loss (<= 0 ROI)', 'Profit (> 0 ROI)'])
plt.tight_layout()
plt.savefig('eda_profit_distribution.png')
plt.close()

print("Success! Check your folder for the 3 new PNG image files.")
