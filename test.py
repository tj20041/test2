import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Glue job initialisation
# ---------------------------------------------------------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ---------------------------------------------------------------------------
# Read source data
# NOTE: Replace these paths / catalogue references with your actual sources.
# ---------------------------------------------------------------------------
orders_df = spark.read.parquet("s3://your-bucket/orders/")
customers_df = spark.read.parquet("s3://your-bucket/customers/")

# ---------------------------------------------------------------------------
# Join — use the list-of-column-names form so PySpark automatically merges
# both sides of the join key into a single output column, eliminating the
# AMBIGUOUS_REFERENCE error that occurs when the join condition is expressed
# as a string equality predicate (e.g. orders_df.customer_id ==
# customers_df.customer_id) which retains duplicate columns in the schema.
# ---------------------------------------------------------------------------
enriched_orders = orders_df.join(customers_df, on=["customer_id"], how="inner")

# ---------------------------------------------------------------------------
# Post-join schema assertion — fail fast with a clear message if duplicate
# column names are detected before any downstream transformation.
# ---------------------------------------------------------------------------
duplicate_cols = [
    c for c in enriched_orders.columns
    if enriched_orders.columns.count(c) > 1
]
assert len(duplicate_cols) == 0, (
    f"Duplicate columns detected after join: {duplicate_cols}. "
    "Use the on=['col'] list syntax or drop redundant join-key columns."
)

# ---------------------------------------------------------------------------
# Select final columns — all references are now unambiguous because the
# list-based join above produced exactly one 'customer_id' column.
# Line 43 in the original failing script was a .select() call on the joined
# DataFrame; the fix above (changing to on=['customer_id']) resolves that.
# ---------------------------------------------------------------------------
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("order_date"),
    F.col("order_amount"),
    F.col("customer_name"),
    F.col("loyalty_tier"),
)

# ---------------------------------------------------------------------------
# Write output to the Gold layer
# NOTE: Replace the output path with your actual Gold-layer S3 destination.
# ---------------------------------------------------------------------------
(
    final_df.write
    .mode("overwrite")
    .parquet("s3://your-bucket/gold/enriched_orders/")
)

job.commit()
