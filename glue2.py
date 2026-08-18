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

# Join orders with customer profiles using the 'using columns' string form so
# that PySpark automatically deduplicates the join key column. Using a
# Column-equality predicate (orders_df.customer_id == customers_df.customer_id)
# retains two identically-named 'customer_id' columns in the output, which
# causes AnalysisException: [AMBIGUOUS_REFERENCE] on any subsequent unqualified
# reference to that column name.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Defensive guard: assert no duplicate column names were introduced by the join.
# This surfaces schema ambiguity at the join site immediately in CloudWatch Logs
# rather than propagating silently to a downstream transformation.
dup_cols = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
if dup_cols:
    raise ValueError(f"Duplicate columns detected after join: {dup_cols}")

# Select final fields for downstream reporting.
# F.col("customer_id") now resolves unambiguously because the join above
# produced exactly one deduplicated 'customer_id' column.
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Process final dataset
final_df.collect()

job.commit()
