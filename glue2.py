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

# Join orders with customer profiles.
# Using a string column name as the join key (on="customer_id") is the
# idiomatic AWS Glue PySpark pattern: PySpark deduplicates the join key
# and retains exactly one 'customer_id' column in the result, preventing
# the AnalysisException: [AMBIGUOUS_REFERENCE] that occurs when the join
# is expressed as a cross-DataFrame equality predicate
# (orders_df.customer_id == customers_df.customer_id), which preserves
# both copies of the column under the same name.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Guard: detect any duplicate column names introduced by schema drift
# before proceeding to the select step, so failures surface with a clear
# message rather than a cryptic AMBIGUOUS_REFERENCE at select time.
duplicate_cols = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
if duplicate_cols:
    raise ValueError(f"Duplicate columns detected after join: {duplicate_cols}")

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
