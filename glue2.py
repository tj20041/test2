import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

try:
    # Join orders with customer profiles using a shared join key.
    # Joining on the column name (instead of orders_df.customer_id ==
    # customers_df.customer_id) collapses both sides' customer_id columns
    # into a single, unambiguous 'customer_id' column in the result.
    enriched_orders = orders_df.join(
        customers_df,
        'customer_id',
        'inner'
    )

    # Defensive check: guarantee there is exactly one 'customer_id' column
    # before it reaches any downstream select(). This surfaces schema-drift
    # or duplicate-column issues clearly instead of a generic
    # GEN-UNCLASSIFIED-ERROR/AnalysisException at select time.
    customer_id_count = enriched_orders.columns.count('customer_id')
    if customer_id_count != 1:
        raise ValueError(
            f"Expected exactly 1 'customer_id' column after join, found "
            f"{customer_id_count}. Schema: {enriched_orders.schema.fieldNames()}"
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

except Exception as e:
    logger.error("Transformation stage failed. Dumping schema for diagnostics.")
    try:
        enriched_orders.printSchema()
    except NameError:
        logger.error("enriched_orders was not created before failure.")
    raise

job.commit()
