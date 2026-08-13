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

# Join orders with customer profiles using string-key form so PySpark
# automatically deduplicates the shared 'customer_id' column, preventing
# an AMBIGUOUS_REFERENCE AnalysisException in the subsequent select().
enriched_orders = orders_df.join(
    customers_df,
    "customer_id",
    "inner"
)

# Guard: assert no duplicate column names exist in the joined DataFrame
# before any select(), withColumn(), or filter() is applied.
join_col_names = [f.name for f in enriched_orders.schema.fields]
assert len(join_col_names) == len(set(join_col_names)), \
    "Duplicate column names detected after join — resolve before select()"

# Select final fields for downstream reporting
final_df = enriched_orders.select(
    F.col("customer_id"),
    F.col("order_id"),
    F.col("customer_name"),
    F.col("order_amount")
)

# Print schema to catch any future schema drift early
final_df.printSchema()

# Process final dataset
final_df.collect()

job.commit()
