import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.errors import AnalysisException

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
# Using the string form of 'on' so PySpark performs a natural equi-join and
# retains only a single deduplicated 'customer_id' column in the output schema,
# eliminating the AMBIGUOUS_REFERENCE AnalysisException that arises when the
# column-object equality form (orders_df.customer_id == customers_df.customer_id)
# is used — which preserves both join-key columns under the same name.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Defensive assertion: fail fast with a clear message if any future schema
# change re-introduces duplicate column names after the join.
assert len(enriched_orders.columns) == len(set(enriched_orders.columns)), (
    f"Duplicate columns detected after join: {enriched_orders.columns}"
)

try:
    # Select final fields for downstream reporting.
    # 'customer_id' is now unambiguous because the string-form join above
    # deduplicates the join key into a single column.
    final_df = enriched_orders.select(
        F.col("customer_id"),
        F.col("order_id"),
        F.col("customer_name"),
        F.col("order_amount")
    )
except AnalysisException as e:
    # Surface schema-level failures with the full enriched schema so the
    # offending duplicate column is immediately visible in the Glue logs.
    print("[ERROR] AnalysisException during select — enriched_orders schema:")
    enriched_orders.printSchema()
    raise

# Process final dataset
final_df.collect()

job.commit()
