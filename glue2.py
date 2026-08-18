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
logger = glueContext.get_logger()
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

# Join orders with customer profiles using string key to perform a USING-style
# join. This de-duplicates the join key column so that only one 'customer_id'
# column is present in the output schema, preventing AMBIGUOUS_REFERENCE errors
# in subsequent select() calls.
enriched_orders = orders_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

# Assert no duplicate column names exist in the joined DataFrame.
# This guard catches any future schema regressions early with a clear message.
duplicate_cols = [
    c for c in enriched_orders.columns
    if enriched_orders.columns.count(c) > 1
]
assert len(duplicate_cols) == 0, (
    f"Duplicate columns detected after join: {duplicate_cols}"
)

# Log joined schema for audit trail in CloudWatch Logs.
logger.info(f"enriched_orders columns: {enriched_orders.columns}")
logger.info(f"enriched_orders dtypes: {enriched_orders.dtypes}")

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
