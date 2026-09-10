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
# Use a string-based join key (rather than a column-expression condition) so
# Spark automatically de-duplicates the shared 'customer_id' column instead
# of retaining two separate 'customer_id' columns in the resulting frame.
# This avoids the AMBIGUOUS_REFERENCE AnalysisException that previously
# occurred when selecting F.col('customer_id') after the join.
enriched_orders = orders_df.join(
    customers_df,
    "customer_id",
    "inner"
)

# Defensive check: fail fast with a clear, actionable error if a future
# schema change reintroduces a duplicate/ambiguous 'customer_id' column,
# instead of letting it surface later as an opaque Spark AnalysisException.
_customer_id_count = enriched_orders.columns.count("customer_id")
assert _customer_id_count == 1, (
    f"Expected exactly one 'customer_id' column after join, found "
    f"{_customer_id_count}. Check join logic in glue2.py for duplicate "
    f"join keys."
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
