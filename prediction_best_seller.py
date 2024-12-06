from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Future Best Seller Prediction with Linear Regression") \
    .getOrCreate()

# Load the average ratings data
input_path = "/Users/saima/Downloads/Project/avg_rating.csv"  # Update with the correct path
data = spark.read.csv(input_path, header=True, inferSchema=True)

# Select relevant columns (remove `average_rating` from features)
data = data.select("id", "productDisplayName", "average_rating")

# Index the categorical column (productDisplayName) for model use
indexer = StringIndexer(inputCol="productDisplayName", outputCol="productIndex")
data = indexer.fit(data).transform(data)

# Assemble features into a single vector excluding `average_rating`
assembler = VectorAssembler(inputCols=["productIndex"], outputCol="features")
data = assembler.transform(data)

# Split the data into training and test sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# Initialize the Linear Regression model with regularization
lr = LinearRegression(featuresCol="features", labelCol="average_rating", regParam=0.1)

# Train the Linear Regression model
lr_model = lr.fit(train_data)

# Make predictions on the test set
predictions = lr_model.transform(test_data)

# Evaluate the model
#evaluator = RegressionEvaluator(labelCol="average_rating", predictionCol="prediction", metricName="rmse")
#rmse = evaluator.evaluate(predictions)
#print(f"Root Mean Square Error (RMSE): {rmse}")

# Show a sample of predictions
predictions.select("productDisplayName", "average_rating", "prediction").show(10)

# Convert predictions to a Pandas DataFrame for visualization
pandas_df = predictions.select("productDisplayName", "average_rating", "prediction").toPandas()

# Sort by actual rating and select top 10 for better visualization
top_predictions_df = pandas_df.sort_values(by="average_rating", ascending=False).head(10)

# Set up the plot style
#sns.set(style="whitegrid")

# Plot the actual vs predicted ratings
#plt.figure(figsize=(14, 8))
#sns.barplot(x="average_rating", y="productDisplayName", data=top_predictions_df, color="blue", label="Actual Rating")
#sns.barplot(x="prediction", y="productDisplayName", data=top_predictions_df, color="orange", alpha=0.6, label="Predicted Rating")
#plt.xlabel("Rating")
#plt.ylabel("Product Name")
#plt.title("Actual vs Predicted Ratings of Top Products")
#plt.legend()
#plt.show()

# Stop Spark session
spark.stop()
