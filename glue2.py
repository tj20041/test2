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

try:
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

    # Join orders with customer profiles using string key so PySpark
    # coalesces both sides of the join key into a single 'customer_id'
    # column, eliminating the AMBIGUOUS_REFERENCE AnalysisException that
    # occurred when using the boolean column-equality expression.
    enriched_orders = orders_df.join(
        customers_df,
        on='customer_id',
        how='inner'
    )

    # Defensive guard: assert no duplicate column names were introduced by
    # the join before proceeding to select, so regressions are caught early.
    dup_cols = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
    assert not dup_cols, f"Duplicate columns detected after join: {dup_cols}"

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

except Exception as e:
    # Log a structured error message so CloudWatch / Glue console marks
    # the job run as FAILED and any configured alarms / EventBridge rules
    # are triggered correctly, then re-raise so Glue sets the job state.
    print(f"[GLUE JOB ERROR] Job failed with exception: {e}")
    raise
