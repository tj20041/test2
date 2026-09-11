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

    # Join orders with customer profiles.
    # Using the column-name form of join() de-duplicates the shared
    # "customer_id" join key automatically, so the resulting DataFrame
    # only contains a single, unambiguous "customer_id" column instead
    # of two identically-named columns (which previously caused Spark's
    # analyzer to throw an AnalysisException on the ambiguous reference).
    enriched_orders = orders_df.join(
        customers_df,
        "customer_id",
        "inner"
    )

    # Select final fields for downstream reporting.
    # "customer_id" is now unambiguous because of the column-name join above.
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
    # Log schemas to aid triage of future schema-collision / ambiguous
    # reference errors before re-raising so the Glue job still fails fast.
    logger.error("Glue job failed during DataFrame transformation: %s", str(e))
    try:
        logger.error("orders_df schema:")
        orders_df.printSchema()
    except Exception:
        logger.error("orders_df was not initialized before failure.")
    try:
        logger.error("customers_df schema:")
        customers_df.printSchema()
    except Exception:
        logger.error("customers_df was not initialized before failure.")
    raise
