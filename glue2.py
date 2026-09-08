import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Configure structured logging so failures appear clearly in CloudWatch Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Using on='customer_id' as a string key instructs PySpark to perform a
# SQL-style USING join, which emits exactly one customer_id column in the
# result and eliminates the AMBIGUOUS_REFERENCE error at its source.
enriched_orders = orders_df.join(
    customers_df,
    on='customer_id',
    how='inner'
)

# Post-join guard: assert no duplicate column names exist in the joined
# DataFrame before any downstream select or transformation is attempted.
duplicate_cols = [c for c in enriched_orders.columns if enriched_orders.columns.count(c) > 1]
if duplicate_cols:
    raise RuntimeError(
        f"Duplicate column name(s) detected after join: {duplicate_cols}. "
        "Review join keys and input DataFrame schemas."
    )

# Select final fields for downstream reporting.
# F.col('customer_id') is now unambiguous because the string-key join
# produces a single merged customer_id column.
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

logger.info("Final schema: %s", final_df.schema.simpleString())
logger.info("Final row count: %d", final_df.count())

# Process final dataset
final_df.collect()

job.commit()
