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

# Join orders with customer profiles using named string key so PySpark
# automatically coalesces the duplicate 'customer_id' columns into one,
# eliminating any AMBIGUOUS_REFERENCE error in downstream operations.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Guard-rail: assert no duplicate column names exist in the joined DataFrame
# before proceeding, so any future join refactoring that re-introduces
# duplicate columns is caught here with a clear, actionable error message.
duplicate_cols = [
    name for name in enriched_orders.columns
    if enriched_orders.columns.count(name) > 1
]
assert len(duplicate_cols) == 0, (
    f"Duplicate columns detected after join: {duplicate_cols}. "
    "Resolve ambiguous column names before selecting."
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
