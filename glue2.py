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
# Using a string key for the join so that PySpark automatically deduplicates
# the shared 'customer_id' column in the output DataFrame, preventing the
# AnalysisException: [AMBIGUOUS_REFERENCE] that occurs when the column-object
# equality form (orders_df.customer_id == customers_df.customer_id) is used,
# because that form retains both copies of 'customer_id' in the joined result.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Guard: assert no duplicate column names survived the join.
# This catches schema drift early, at the join site, rather than at a
# downstream .select() or .filter() call.
assert len(enriched_orders.columns) == len(set(enriched_orders.columns)), (
    f"Duplicate columns detected after join: {enriched_orders.columns}"
)

# Select final fields for downstream reporting.
# After the string-key join above, enriched_orders contains exactly one
# 'customer_id' column, so all references below are unambiguous.
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Process final dataset
final_df.collect()

job.commit()
