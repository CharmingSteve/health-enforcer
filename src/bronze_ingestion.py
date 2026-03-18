"""
bronze_ingestion.py
-------------------
Medallion Architecture – Bronze Layer

Reads the raw Apple Health XML export directly from a Databricks Volume using the spark-xml
library, attaches an ingestion timestamp, and writes the result to a
Bronze Delta table partitioned by `type`.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_PATH: str = "/Volumes/workspace/default/apple-data-march-8-2026/export.xml"
BRONZE_TABLE_PATH: str = "/Volumes/workspace/default/apple-data-march-8-2026/bronze_health_records"
ROW_TAG: str = "Record"

# ---------------------------------------------------------------------------
# Step 1 – Read raw XML from Volume
# ---------------------------------------------------------------------------
def read_apple_health_xml(spark: SparkSession, source_path: str = SOURCE_PATH) -> DataFrame:
    return (
        spark.read.format("xml")
        .option("rowTag", ROW_TAG)
        .load(source_path)
    )

# ---------------------------------------------------------------------------
# Step 2 – Enrich with ingestion metadata
# ---------------------------------------------------------------------------
def add_ingestion_timestamp(df: DataFrame) -> DataFrame:
    return df.withColumn("ingestion_timestamp", F.current_timestamp())

# ---------------------------------------------------------------------------
# Step 3 – Persist to Bronze Delta table
# ---------------------------------------------------------------------------
def write_bronze_table(
    df: DataFrame,
    output_path: str = BRONZE_TABLE_PATH,
    partition_col: str = "_type",
) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .partitionBy(partition_col)
        .save(output_path)
    )

# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------
def ingest_bronze(spark: SparkSession) -> DataFrame:
    raw_df = read_apple_health_xml(spark)
    enriched_df = add_ingestion_timestamp(raw_df)
    write_bronze_table(enriched_df)
    return enriched_df