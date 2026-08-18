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
# Use the 'using column name' string syntax so Spark automatically coalesces
# the shared key into a single 'customer_id' output column, eliminating the
# AMBIGUOUS_REFERENCE AnalysisException that arises when both sides of the
# join carry an identically-named column and F.col() is used unqualified.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Assert no duplicate column names exist in the joined DataFrame before
# proceeding to the select, so any future schema-drift regressions surface
# at the join boundary rather than further downstream.
assert len(set(enriched_orders.columns)) == len(enriched_orders.columns), (
    "Duplicate column names detected in enriched_orders after join: "
    + str(enriched_orders.columns)
)

# Select final fields for downstream reporting.
# All four columns are now unambiguous: 'customer_id' is coalesced by the
# using-syntax join; 'order_id', 'order_amount', and 'customer_name' each
# exist on only one side of the join.
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Process final dataset
final_df.collect()

job.commit()
