import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df_sales = spark.read.table("sales_orders")
df_customers = spark.read.table("customer_master")
df_products = spark.read.table("product_catalog")

sales_filtered = df_sales.filter(col("order_date") >= "2026-01-01")
customers_filtered = df_customers.select("customer_id", "customer_name", "segment", "region", "customer_id")
products_filtered = df_products.select("product_id", "product_name", "category")

joined_df = sales_filtered.join(customers_filtered, "customer_id").join(products_filtered, sales_filtered.product_id == products_filtered.product_id)

joined_df.write.mode("overwrite").parquet("s3://output-bucket/sales_enriched/")
job.commit()
