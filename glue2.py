import sys
from collections import Counter
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
# coalesces the shared join column and emits only a single customer_id
# in the output schema, preventing AMBIGUOUS_REFERENCE downstream.
enriched_orders = orders_df.join(
    customers_df,
    "customer_id",
    "inner"
)

# Defensive post-join schema guard: fail fast with a clear message if
# future schema changes introduce duplicate column names after the join.
duplicate_cols = [c for c, count in Counter(enriched_orders.columns).items() if count > 1]
if duplicate_cols:
    raise ValueError(
        f"Duplicate columns detected in enriched_orders after join: {duplicate_cols}. "
        "Check that neither orders_df nor customers_df introduces overlapping column names."
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
