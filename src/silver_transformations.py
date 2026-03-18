"""
silver_transformations.py
--------------------------
Medallion Architecture – Silver Layer

Contains pure PySpark transformation functions that clean and enrich the
Bronze health records.  Every function accepts a DataFrame and returns a
DataFrame so it can be composed freely and tested in isolation.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

# Record types that are too noisy / low-signal for downstream ML use.
_NOISE_TYPES: list[str] = [
    "HeadphoneAudioExposure",
    "WalkingAsymmetryPercentage",
]


def filter_noise(df: DataFrame) -> DataFrame:
    """Remove low-signal record types from the dataset.

    Drops rows whose ``type`` column matches any entry in ``_NOISE_TYPES``
    (currently ``HeadphoneAudioExposure`` and ``WalkingAsymmetryPercentage``).

    Args:
        df: Input DataFrame containing at least a ``type`` column.

    Returns:
        DataFrame with noisy record types removed.
    """
    return df.filter(~F.col("type").isin(_NOISE_TYPES))


# ---------------------------------------------------------------------------
# Shabbat imputation
# ---------------------------------------------------------------------------

# Day-of-week constant: Saturday = 7 in PySpark's ``dayofweek`` (Sun=1 … Sat=7).
_SATURDAY: int = 7

# Hour-of-day range (inclusive) that represents mid-morning Shabbat rest.
_SHABBAT_HOUR_START: int = 10
_SHABBAT_HOUR_END: int = 12

# Default imputed energy value (kcal) to represent minimal activity.
_SHABBAT_IMPUTE_VALUE: float = 400.0


def impute_shabbat_gap(df: DataFrame) -> DataFrame:
    """Fill missing ``ActiveEnergyBurned`` values during the Shabbat rest window.

    Apple Watch stops recording during Shabbat observance, leaving gaps in
    ``ActiveEnergyBurned``.  This function imputes those gaps with a
    physiologically plausible resting value (400 kcal) when:

    * the row's ``startDate`` falls on a **Saturday** (day-of-week == 7), AND
    * the **hour** of ``startDate`` is between 10 and 12 (inclusive), AND
    * ``ActiveEnergyBurned`` is currently ``null``.

    Args:
        df: Input DataFrame containing ``startDate`` (TimestampType) and
            ``ActiveEnergyBurned`` (DoubleType or compatible numeric) columns.

    Returns:
        DataFrame with eligible null ``ActiveEnergyBurned`` values replaced
        by ``400.0``.
    """
    is_saturday = F.dayofweek(F.col("startDate")) == _SATURDAY
    is_shabbat_hour = F.hour(F.col("startDate")).between(
        _SHABBAT_HOUR_START, _SHABBAT_HOUR_END
    )
    is_null_energy = F.col("ActiveEnergyBurned").isNull()

    shabbat_condition = is_saturday & is_shabbat_hour & is_null_energy

    return df.withColumn(
        "ActiveEnergyBurned",
        F.when(shabbat_condition, F.lit(_SHABBAT_IMPUTE_VALUE)).otherwise(
            F.col("ActiveEnergyBurned")
        ),
    )
