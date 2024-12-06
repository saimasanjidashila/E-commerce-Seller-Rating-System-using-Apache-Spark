import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into a Pandas DataFrame
input_path = "/Users/saima/Downloads/Project/avg_rating.csv"
pandas_df = pd.read_csv(input_path)

# Sort and select the top 10 products by average rating
top_sellers_df = pandas_df.sort_values(by="average_rating", ascending=False).head(10)

# Set the style for the plot
sns.set(style="whitegrid")

# Customize the color palette
palette = sns.color_palette("magma", n_colors=10)

# Plot the data - top 10 sellers by average rating with labels
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x="average_rating", y="productDisplayName", data=top_sellers_df, palette=palette)
plt.xlabel("Average Rating", fontsize=14)
plt.ylabel("Product Name", fontsize=14)
plt.title("Top 10 Best Sellers by Average Rating", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis for ranking

# Add value labels to each bar
for index, value in enumerate(top_sellers_df["average_rating"]):
    plt.text(value, index, f'{value:.0f}', va='center', ha='left', fontsize=12)

plt.show()
