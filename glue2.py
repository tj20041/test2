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
# Using a string key for the 'on' argument so that PySpark automatically
# deduplicates the join column, producing exactly one 'customer_id' in
# the resulting DataFrame and avoiding AMBIGUOUS_REFERENCE on .select().
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Defensive guard: assert no duplicate column names exist after the join.
# This will surface schema collisions immediately at join time rather than
# with an opaque AnalysisException at a later .select() or action.
duplicate_cols = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
assert len(duplicate_cols) == 0, f"Duplicate columns detected after join: {enriched_orders.columns}"

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
