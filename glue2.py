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
# Use the single-column join-key syntax ('customer_id') instead of the
# equality-expression form (orders_df.customer_id == customers_df.customer_id).
# This automatically de-duplicates the join key column in the result set,
# preventing the ambiguous column reference AnalysisException that occurs
# when both DataFrames retain their own copy of a column with the same name.
enriched_orders = orders_df.join(
    customers_df,
    'customer_id',
    'inner'
)

# Defensive validation: fail fast with a clear, actionable message if any
# future change to this join (or the source schemas) reintroduces duplicate
# column names, rather than surfacing only as a generic Glue AnalysisException.
column_names = enriched_orders.columns
duplicate_columns = {name for name in column_names if column_names.count(name) > 1}
if duplicate_columns:
    raise ValueError(
        "Ambiguous/duplicate column(s) detected after join in glue2.py: "
        f"{sorted(duplicate_columns)}. Alias the source DataFrames or drop "
        "the redundant column(s) before selecting downstream fields."
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

job.commit()
