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
# FIX: Use a string column name as the join key (on='customer_id') instead of
# the column-object equality form (orders_df.customer_id == customers_df.customer_id).
# When a string key is passed, PySpark automatically deduplicates the join key
# column and retains only one 'customer_id' in the output DataFrame, eliminating
# the AMBIGUOUS_REFERENCE AnalysisException that occurred at the subsequent select().
enriched_orders = orders_df.join(
    customers_df,
    on='customer_id',
    how='inner'
)

# Guard: assert no duplicate column names exist after the join.
# This catches schema regressions early if additional joins are added in future.
duplicate_cols = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
assert len(duplicate_cols) == 0, (
    f"[FATAL] Duplicate columns detected after join: {duplicate_cols}. "
    "Use string-key join (on='col_name') or qualify references with DataFrame aliases."
)

# Select final fields for downstream reporting.
# With the string-key join fix applied above, 'customer_id' is unambiguous here.
try:
    final_df = enriched_orders.select(
        F.col("customer_id"),
        F.col("order_id"),
        F.col("customer_name"),
        F.col("order_amount")
    )
except Exception as e:
    print(f"[FATAL] Failed to select final columns from enriched_orders: {e}")
    sys.exit(1)

# Process final dataset
try:
    final_df.collect()
except Exception as e:
    print(f"[FATAL] Failed to collect final_df: {e}")
    sys.exit(1)

job.commit()
