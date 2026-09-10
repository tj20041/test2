import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger = glueContext.get_logger()

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

# Join orders with customer profiles using the join-key-list form so that
# the shared 'customer_id' column is automatically deduped into a single,
# unambiguous column instead of retaining two physically distinct columns
# named 'customer_id' (one from each side of the join).
enriched_orders = orders_df.join(
    customers_df,
    "customer_id",
    "inner"
)

# Defensive schema check: fail fast with a clear, actionable message if a
# future change to the join reintroduces a duplicate 'customer_id' column,
# rather than letting Spark raise a generic AMBIGUOUS_REFERENCE
# AnalysisException deep inside the select() call.
customer_id_count = enriched_orders.columns.count("customer_id")
if customer_id_count != 1:
    error_message = (
        f"Expected exactly one 'customer_id' column after join, found "
        f"{customer_id_count}. Columns: {enriched_orders.columns}"
    )
    logger.error(error_message)
    raise ValueError(error_message)

# Select final fields for downstream reporting
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Process final dataset
try:
    final_df.collect()
except AnalysisException as e:
    logger.error(f"AnalysisException while processing final_df: {e}")
    raise

job.commit()
