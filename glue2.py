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
    # Use the single-column join-key form so Spark automatically
    # deduplicates the shared 'customer_id' column instead of
    # leaving two ambiguous columns with the same name in the result.
    enriched_orders = orders_df.join(
        customers_df,
        "customer_id",
        "inner"
    )

    # Select final fields for downstream reporting.
    # 'customer_id' is now unambiguous because the join above
    # already collapsed the duplicate columns.
    final_df = enriched_orders.select(
        F.col("customer_id"),
        F.col("order_id"),
        F.col("customer_name"),
        F.col("order_amount")
    )

    # Process final dataset
    final_df.collect()

    job.commit()

except AnalysisException as e:
    logger.error(
        "Spark analysis error while processing orders/customers join. "
        "orders_df schema: {0} | customers_df schema: {1} | error: {2}".format(
            orders_df.schema.simpleString() if 'orders_df' in dir() else 'N/A',
            customers_df.schema.simpleString() if 'customers_df' in dir() else 'N/A',
            str(e)
        )
    )
    raise
except Exception as e:
    logger.error("Unexpected error in Glue job '{0}': {1}".format(args['JOB_NAME'], str(e)))
    raise
