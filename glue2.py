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
# Using a string key for 'on' so PySpark automatically deduplicates the
# shared 'customer_id' column in the output DataFrame, preventing the
# AnalysisException: AMBIGUOUS_REFERENCE error that occurs when a
# column-equality expression (df1.col == df2.col) is used and both source
# columns are retained under the same name.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Defensive assertion: catch any duplicate column names immediately after
# the join so that schema bugs are surfaced at development time rather than
# as a runtime AnalysisException in production.
assert len(enriched_orders.columns) == len(set(enriched_orders.columns)), (
    f"Duplicate columns detected after join: {enriched_orders.columns}"
)

# Select final fields for downstream reporting.
# 'customer_id' is now unambiguous because the string-key join above emits
# only a single deduplicated column.
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Process final dataset
final_df.collect()

job.commit()
