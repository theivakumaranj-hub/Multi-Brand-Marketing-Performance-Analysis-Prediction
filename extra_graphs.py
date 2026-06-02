import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv('cleaned_campaign_data.csv')

print("Generating new visual graphs...")

# ---------------------------------------------------------
# GRAPH 1: Top 5 Most Profitable Campaigns (Matches SQL Query 1)
# ---------------------------------------------------------
top_5_roi = df.sort_values(by='ROI', ascending=False).head(5)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_5_roi, x='Campaign_ID', y='ROI', hue='Campaign_ID', palette='magma', legend=False)
plt.title('Top 5 Most Profitable Campaigns by ROI')
plt.ylabel('Return on Investment (%)')
plt.xlabel('Campaign ID')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sql_graph_top_5_roi.png')
plt.close()

# ---------------------------------------------------------
# GRAPH 2: Average Revenue by Target Audience (Matches SQL Query 3)
# ---------------------------------------------------------
avg_revenue = df.groupby('Target_Audience')['Revenue'].mean().reset_index()
avg_revenue = avg_revenue.sort_values(by='Revenue', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_revenue, x='Target_Audience', y='Revenue', hue='Target_Audience', palette='viridis', legend=False)
plt.title('Average Revenue per Target Audience')
plt.ylabel('Average Revenue (₹)')
plt.xlabel('Target Audience')
plt.tight_layout()
plt.savefig('sql_graph_avg_revenue.png')
plt.close()

print("Success! Check your folder for 'sql_graph_top_5_roi.png' and 'sql_graph_avg_revenue.png'")