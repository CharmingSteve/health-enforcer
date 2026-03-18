"""
bronze_ingestion.py
-------------------
Medallion Architecture – Bronze Layer

Reads the raw Apple Health XML export from AWS S3 using the spark-xml
library, attaches an ingestion timestamp, and writes the result to a
Bronze Delta table partitioned by `type`.

All logic is expressed as pure functions that accept and return PySpark
DataFrames so they can be unit-tested independently of a live cluster.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_PATH: str = "/Volumes/workspace/default/apple-data-march-8-2026/export.xml"
BRONZE_TABLE_PATH: str = "/Volumes/workspace/default/bronze_health_records"
ROW_TAG: str = "Record"


# ---------------------------------------------------------------------------
# Step 1 – Read raw XML from S3
# ---------------------------------------------------------------------------
def read_apple_health_xml(spark: SparkSession, source_path: str = SOURCE_PATH) -> DataFrame:
    """Read the Apple Health XML export and return a raw DataFrame.

    Uses the ``com.databricks.spark.xml`` (spark-xml) reader with
    ``rowTag=Record`` so that every ``<Record>`` element becomes one row.

    Args:
        spark: Active :class:`~pyspark.sql.SparkSession`.
        source_path: ``s3a://`` URI pointing to the XML file.

    Returns:
        Raw PySpark DataFrame with one row per ``<Record>`` element.
    """
    return (
        spark.read.format("xml")
        .option("rowTag", ROW_TAG)
        .load(source_path)
    )


# ---------------------------------------------------------------------------
# Step 2 – Enrich with ingestion metadata
# ---------------------------------------------------------------------------
def add_ingestion_timestamp(df: DataFrame) -> DataFrame:
    """Append a UTC ingestion timestamp to the DataFrame.

    Args:
        df: Input DataFrame (typically the raw bronze read).

    Returns:
        DataFrame with an additional ``ingestion_timestamp`` column of type
        ``TimestampType``.
    """
    return df.withColumn("ingestion_timestamp", F.current_timestamp())


# ---------------------------------------------------------------------------
# Step 3 – Persist to Bronze Delta table
# ---------------------------------------------------------------------------
def write_bronze_table(
    df: DataFrame,
    output_path: str = BRONZE_TABLE_PATH,
    partition_col: str = "type",
) -> None:
    """Write the enriched DataFrame to the Bronze Delta table.

    The table is partitioned by ``type`` (e.g. ``HKQuantityTypeIdentifierStepCount``)
    to allow efficient downstream reads per record type.

    Args:
        df: Enriched DataFrame to persist.
        output_path: ``s3a://`` URI for the Delta table root directory.
        partition_col: Column name to use for partitioning (default: ``type``).
    """
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
    """End-to-end Bronze ingestion: read → enrich → write.

    This function composes the individual steps and is the primary
    entry-point used by Databricks jobs.

    Args:
        spark: Active :class:`~pyspark.sql.SparkSession`.

    Returns:
        The enriched Bronze DataFrame (after the write).
    """
    raw_df = read_apple_health_xml(spark)
    enriched_df = add_ingestion_timestamp(raw_df)
    write_bronze_table(enriched_df)
    return enriched_df
