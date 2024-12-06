from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Seller Rating System") \
    .getOrCreate()

# Load cleaned datasets
customer_df = spark.read.csv("/Users/saima/Downloads/Project/customer.csv", header=True, inferSchema=True)
product_df = spark.read.csv("/Users/saima/Downloads/Project/product.csv", header=True, inferSchema=True)
transaction_df = spark.read.csv("/Users/saima/Downloads/Project/transactions.csv", header=True, inferSchema=True)

# Join DataFrames to get product information with transactions
successful_transactions = transaction_df.filter(transaction_df.payment_status == "Success")

joined_df = successful_transactions \
    .join(product_df, successful_transactions.product_metadata.contains(product_df.id.cast("string")), "inner") \
    .groupBy("id", "productDisplayName") \
    .agg({"total_amount": "avg"}) \
    .withColumnRenamed("avg(total_amount)", "average_rating")

# Show final seller rating analysis
joined_df.show()
output_path = "/Users/saima/Downloads/Project/average_ratings.csv"  # Replace with your desired path
joined_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
spark.stop()