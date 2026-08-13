import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'output_path'])
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
# Using a string join key instead of a boolean column expression so that
# Spark automatically deduplicates the shared 'customer_id' column and
# leaves exactly one 'customer_id' in the resulting DataFrame, preventing
# the AMBIGUOUS_REFERENCE AnalysisException that occurred with the
# previous boolean-expression join form.
enriched_orders = orders_df.join(
    customers_df,
    "customer_id",
    "inner"
)

# Post-join guard: assert no duplicate column names exist before any
# downstream select/transform step. Catches this class of error early
# in development and CI without impacting production performance.
dupes = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
assert not dupes, f"Duplicate columns detected after join: {dupes}"

# Select final fields for downstream reporting
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Write output via GlueContext to the configured S3 output path.
# Using a DynamicFrame write instead of collect() avoids pulling the
# entire dataset into driver memory, which would cause OOM failures at
# production data volumes.
output_dyf = DynamicFrame.fromDF(final_df, glueContext, "output")
glueContext.write_dynamic_frame.from_options(
    frame=output_dyf,
    connection_type="s3",
    connection_options={"path": args["output_path"], "partitionKeys": []},
    format="parquet"
)

job.commit()
