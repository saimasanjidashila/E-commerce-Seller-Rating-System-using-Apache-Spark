import pandas as pd
import matplotlib.pyplot as plt

# Load transaction data
transaction_df = pd.read_csv("/Users/saima/Downloads/Project/transactions.csv")

# Convert 'created_at' to datetime format
transaction_df['created_at'] = pd.to_datetime(transaction_df['created_at'])

# Calculate monthly sales trend
sales_trend = transaction_df.set_index('created_at').resample('M')['total_amount'].sum()

# Plot monthly sales trend
plt.figure(figsize=(14, 7))
sales_trend.plot(color="skyblue")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales Amount")
plt.show()
