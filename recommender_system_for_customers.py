from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Recommender System for Customers") \
    .getOrCreate()

# Load transaction data with relevant columns
transaction_df = spark.read.csv("/Users/saima/Downloads/Project/transactions.csv", header=True, inferSchema=True)

# Extract the product_id from the JSON-like structure in product_metadata
# Assuming product_metadata is structured like [{'product_id': '12345', ...}]
# Using get_json_object to parse JSON string to extract the product_id
transaction_df = transaction_df.withColumn("product_id", get_json_object("product_metadata", "$[0].product_id").cast("integer"))

# Verify if product_id has been extracted
#transaction_df.select("product_metadata", "product_id").show(10, truncate=False)

# Filter and prepare the data for ALS (remove any nulls from key columns)
als_data = transaction_df.select(
    col("customer_id").cast("integer").alias("userId"),
    col("product_id").alias("itemId"),
    col("total_amount").cast("float").alias("rating")
).filter(col("userId").isNotNull() & col("itemId").isNotNull() & col("rating").isNotNull())

# Check the ALS data count
print(f"Count of ALS data after filtering: {als_data.count()}")
#als_data.show(10)

# Ensure ALS data is not empty before proceeding
if als_data.count() > 0:
    # Split the data into training and test sets
    (training_data, test_data) = als_data.randomSplit([0.8, 0.2], seed=42)

    # Initialize ALS model
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    # Train the ALS model
    als_model = als.fit(training_data)

    # Make predictions on the test set
    predictions = als_model.transform(test_data)

    # Evaluate the model
    #evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    #rmse = evaluator.evaluate(predictions)
    #print(f"Root-mean-square error = {rmse}")

    # Generate top 5 product recommendations for each user
    user_recommendations = als_model.recommendForAllUsers(5)
    user_recommendations.show(truncate=False)
else:
    print("ALS data is empty. Check your input data for issues.")

# Stop the Spark session
spark.stop()
