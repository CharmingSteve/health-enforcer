"""
tests/test_silver_transformations.py
--------------------------------------
Unit tests for silver_transformations.py.

Uses a local SparkSession (no cluster required) to create small mock
DataFrames and validate the transformation logic in isolation.
"""

import sys
import os

# Allow importing from src/ without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from silver_transformations import filter_noise, impute_shabbat_gap


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark():
    """Provide a local SparkSession shared across the test session."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("health-enforcer-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Schema shared by Silver-layer tests
# ---------------------------------------------------------------------------

_SILVER_SCHEMA = StructType(
    [
        StructField("type", StringType(), nullable=True),
        StructField("startDate", TimestampType(), nullable=True),
        StructField("ActiveEnergyBurned", DoubleType(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Tests for filter_noise
# ---------------------------------------------------------------------------


class TestFilterNoise:
    def test_removes_headphone_audio_exposure(self, spark):
        data = [
            ("HeadphoneAudioExposure", None, None),
            ("StepCount", None, 100.0),
        ]
        df = spark.createDataFrame(data, schema=_SILVER_SCHEMA)
        result = filter_noise(df)
        types = [row["type"] for row in result.collect()]
        assert "HeadphoneAudioExposure" not in types

    def test_removes_walking_asymmetry_percentage(self, spark):
        data = [
            ("WalkingAsymmetryPercentage", None, None),
            ("StepCount", None, 100.0),
        ]
        df = spark.createDataFrame(data, schema=_SILVER_SCHEMA)
        result = filter_noise(df)
        types = [row["type"] for row in result.collect()]
        assert "WalkingAsymmetryPercentage" not in types

    def test_preserves_valid_record_types(self, spark):
        data = [
            ("StepCount", None, 100.0),
            ("ActiveEnergyBurned", None, 200.0),
        ]
        df = spark.createDataFrame(data, schema=_SILVER_SCHEMA)
        result = filter_noise(df)
        assert result.count() == 2

    def test_both_noise_types_removed_together(self, spark):
        data = [
            ("HeadphoneAudioExposure", None, None),
            ("WalkingAsymmetryPercentage", None, None),
            ("HeartRate", None, 72.0),
        ]
        df = spark.createDataFrame(data, schema=_SILVER_SCHEMA)
        result = filter_noise(df)
        assert result.count() == 1
        assert result.first()["type"] == "HeartRate"


# ---------------------------------------------------------------------------
# Tests for impute_shabbat_gap
# ---------------------------------------------------------------------------


class TestImputeShabbatGap:
    def _make_row(self, spark, timestamp_str: str, energy):
        """Helper: create a single-row DataFrame for a given timestamp."""
        data = [("ActiveEnergyBurned", timestamp_str, energy)]
        schema = StructType(
            [
                StructField("type", StringType(), nullable=True),
                StructField("startDate_str", StringType(), nullable=True),
                StructField("ActiveEnergyBurned", DoubleType(), nullable=True),
            ]
        )
        df = spark.createDataFrame(data, schema=schema)
        return df.withColumn(
            "startDate", F.to_timestamp(F.col("startDate_str"), "yyyy-MM-dd HH:mm:ss")
        ).drop("startDate_str")

    # ------------------------------------------------------------------
    # Primary requirement: null on Saturday at 11:00 → imputed to 400.0
    # ------------------------------------------------------------------

    def test_imputes_null_on_saturday_at_11am(self, spark):
        """Core test: null ActiveEnergyBurned on Saturday 11 AM → 400.0."""
        # 2026-03-14 is a Saturday.
        df = self._make_row(spark, "2026-03-14 11:00:00", None)
        result = impute_shabbat_gap(df)
        value = result.first()["ActiveEnergyBurned"]
        assert value == 400.0, f"Expected 400.0, got {value}"

    # ------------------------------------------------------------------
    # Boundary: hour at edge of the Shabbat window
    # ------------------------------------------------------------------

    def test_imputes_null_on_saturday_at_10am(self, spark):
        df = self._make_row(spark, "2026-03-14 10:00:00", None)
        result = impute_shabbat_gap(df)
        assert result.first()["ActiveEnergyBurned"] == 400.0

    def test_imputes_null_on_saturday_at_noon(self, spark):
        df = self._make_row(spark, "2026-03-14 12:00:00", None)
        result = impute_shabbat_gap(df)
        assert result.first()["ActiveEnergyBurned"] == 400.0

    # ------------------------------------------------------------------
    # Negative cases: must NOT impute
    # ------------------------------------------------------------------

    def test_does_not_impute_outside_hour_window(self, spark):
        """Saturday but outside 10–12 → no imputation."""
        df = self._make_row(spark, "2026-03-14 09:00:00", None)
        result = impute_shabbat_gap(df)
        assert result.first()["ActiveEnergyBurned"] is None

    def test_does_not_impute_on_weekday(self, spark):
        """Tuesday in the same hour range → no imputation."""
        # 2026-03-10 is a Tuesday.
        df = self._make_row(spark, "2026-03-10 11:00:00", None)
        result = impute_shabbat_gap(df)
        assert result.first()["ActiveEnergyBurned"] is None

    def test_does_not_overwrite_existing_value(self, spark):
        """Saturday at 11 AM but value already present → unchanged."""
        df = self._make_row(spark, "2026-03-14 11:00:00", 250.0)
        result = impute_shabbat_gap(df)
        assert result.first()["ActiveEnergyBurned"] == 250.0
