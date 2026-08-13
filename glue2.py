import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

# ==========================================
# 0. GLUE INITIALIZATION & LOGGING SETUP
# ==========================================
# Fetch job name passed by the AWS Glue execution environment
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger = logging.getLogger("SalesOrderProcessingETL")
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("Initializing Sales Order Processing ETL Job...")

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    orders_schema = StructType([
        StructField("order_id", IntegerType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("order_status", StringType(), True)
    ])

    customers_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("customer_name", StringType(), True),
        StructField("loyalty_tier", StringType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting data into Bronze layer...")
    orders_bronze = spark.createDataFrame([
        (101, 1, 250.0, "COMPLETED"),
        (102, 2, 80.0, "CANCELLED"),
        (103, 3, 420.5, "COMPLETED"),
        (104, 1, 15.0, "COMPLETED")
    ], schema=orders_schema)

    customers_bronze = spark.createDataFrame([
        (1, "Alice Smith", "GOLD"),
        (2, "Bob Jones", "BRONZE"),
        (3, "Charlie Brown", "PLATINUM")
    ], schema=customers_schema)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering valid orders for Silver layer...")
    orders_silver = orders_bronze.filter("order_status = 'COMPLETED'")
    customers_silver = customers_bronze.filter("loyalty_tier IS NOT NULL")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Enriching orders with customer metadata for Gold layer...")

    # SPECIFIC ERROR: Ambiguous column reference in join
    # Joining with binary equality (orders_silver.customer_id == customers_silver.customer_id)
    # preserves duplicate 'customer_id' columns in the resulting dataframe, causing PySpark's
    # .select("customer_id", ...) to fail with an AnalysisException: Reference 'customer_id' is ambiguous.
    # FIX: Use on="customer_id" (or select orders_silver["customer_id"])
    joined_df = orders_silver.join(
        customers_silver,
        orders_silver.customer_id == customers_silver.customer_id,
        "inner"
    )

    df_gold = joined_df.select(
        "order_id",
        "customer_id",
        "customer_name",
        "loyalty_tier",
        "amount"
    )

    logger.info("Pipeline completed successfully.")
    df_gold.show(truncate=False)

    # Commit Glue Job
    job.commit()

except Exception as e:
    # Error handling and logging for production diagnostics
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
