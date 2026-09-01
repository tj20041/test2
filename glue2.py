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
# Using a string key name instead of a column equality expression ensures
# PySpark deduplicates the join key into a single 'customer_id' column in
# the output DataFrame, preventing an AMBIGUOUS_REFERENCE AnalysisException
# on the subsequent select.
enriched_orders = orders_df.join(
    customers_df,
    'customer_id',
    'inner'
)

# Assert that exactly one 'customer_id' column exists after the join.
# This guard catches any future regression where duplicate columns could
# be reintroduced (e.g. if the join condition is later changed).
assert len([f.name for f in enriched_orders.schema.fields if f.name == 'customer_id']) == 1, \
    'Duplicate customer_id columns detected post-join'

# Select final fields for downstream reporting.
# Using string column names is idiomatic for straightforward column
# selection in Glue PySpark scripts and avoids unnecessary F.col() wrappers.
final_df = enriched_orders.select(
    'customer_id',
    'order_id',
    'customer_name',
    'order_amount'
)

# Process final dataset
final_df.collect()

job.commit()
