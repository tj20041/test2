import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read orders data
orders_df = spark.createDataFrame(
    [
        ("ORD-501", 1001, 150.50, "2026-08-10"),
        ("ORD-502", 1002, 89.99, "2026-08-11"),
    ],
    ["order_id", "customer_id", "order_amount", "order_date"]
)

# Read customer details data
customers_df = spark.createDataFrame(
    [
        (1001, "Acme Corp", "Enterprise"),
        (1002, "Beta LLC", "SMB"),
    ],
    ["customer_id", "customer_name", "segment"]
)

# Join orders with customer profiles using a string key so that PySpark
# automatically deduplicates the shared join column, eliminating any
# AMBIGUOUS_REFERENCE when the column is referenced later by name.
enriched_orders = orders_df.join(
    customers_df,
    "customer_id",
    "inner"
)

# Guard: assert that all expected columns are present in the joined DataFrame
# before attempting the select, so any future schema regression fails fast
# with a clear message rather than an opaque Spark AnalysisException.
expected_cols = {"customer_id", "order_id", "customer_name", "order_amount"}
actual_cols = set(enriched_orders.columns)
assert expected_cols.issubset(actual_cols), (
    f"Missing columns after join: {expected_cols - actual_cols}"
)

# Select final fields for downstream reporting
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Process final dataset
final_df.collect()

job.commit()
