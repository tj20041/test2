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

    # Join orders with customer profiles using string key to avoid duplicate
    # 'customer_id' columns in the output DataFrame. Using a column-object
    # equality expression (orders_df.customer_id == customers_df.customer_id)
    # retains both source columns in the result, causing AMBIGUOUS_REFERENCE
    # when the column is referenced by unqualified name in a downstream select.
    # Passing the join key as a string causes PySpark to coalesce the two
    # join-key columns into a single deduplicated output column.
    enriched_orders = orders_df.join(
        customers_df,
        on='customer_id',
        how='inner'
    )

    # Defensive assertion: ensure exactly one 'customer_id' column exists
    # after the join before proceeding to select.
    duplicate_check = [f.name for f in enriched_orders.schema.fields if f.name == 'customer_id']
    assert len(duplicate_check) == 1, (
        f"Expected exactly 1 'customer_id' column after join, found {len(duplicate_check)}. "
        f"Schema: {enriched_orders.schema.simpleString()}"
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
    # Log the schema of enriched_orders if it exists to aid diagnosis,
    # then re-raise so the Glue job is correctly marked as failed.
    try:
        enriched_orders.printSchema()
    except NameError:
        pass
    raise e

job.commit()
