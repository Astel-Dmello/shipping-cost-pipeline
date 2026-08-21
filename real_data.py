"""
Loader/adapter for a REAL public logistics dataset: the Supply Chain Management System
(SCMS) Delivery History Dataset (USAID-administered health-commodity shipments,
~10,324 rows, 2006-2015).

This is NOT the paper's own dataset (which is private). It is used as a real-world
validation dataset for the pipeline.

Real-world handling:
  - Numeric columns containing placeholder text are coerced to NaN.
  - High-cardinality categorical columns are capped to top N levels + "Other".
  - Date columns are converted into shipment-planning features.
  - Additional numeric features are engineered without using the target variable.
"""
from typing import List
import os
import urllib.request

import numpy as np
import pandas as pd

from feature_selection import TARGET
from logging_config import get_logger

logger = get_logger(__name__)


REAL_DATA_URL = (
    "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/"
    "master/SCMS_Delivery_History_Dataset.csv"
)


# Columns retained from the original SCMS dataset.
RAW_COLUMN_MAP = {
    "Freight Cost (USD)": TARGET,

    # Numeric shipment/product features
    "Weight (Kilograms)": "freight_weight",
    "Pack Price": "pack_price",
    "Unit Price": "unit_price",
    "Line Item Quantity": "line_item_quantity",
    "Line Item Value": "line_item_value",
    "Line Item Insurance (USD)": "line_item_insurance",
    "Unit of Measure (Per Pack)": "unit_of_measure",

    # Categorical features
    "Country": "country",
    "Managed By": "managed_by",
    "Fulfill Via": "fulfill_via",
    "Vendor INCO Term": "vendor_inco_term",
    "Shipment Mode": "shipment_mode",
    "Product Group": "product_group",
    "Sub Classification": "sub_classification",
    "Vendor": "vendor",
    "Item Description": "item_description",
    "Molecule/Test Type": "molecule_test_type",
    "Brand": "brand",
    "Dosage": "dosage",
    "Dosage Form": "dosage_form",
    "Manufacturing Site": "manufacturing_site",
    "First Line Designation": "first_line_designation",
}


# Date columns used to create features.
# We intentionally use planning-time dates rather than Delivered/Recorded dates
# to reduce the risk of using information that may only be known after shipment.
DATE_COLUMN_MAP = {
    "PQ First Sent to Client Date": "pq_sent_date",
    "PO Sent to Vendor Date": "po_sent_date",
    "Scheduled Delivery Date": "scheduled_delivery_date",
}


NUMERIC_COLUMNS = [
    TARGET,
    "freight_weight",
    "pack_price",
    "unit_price",
    "line_item_quantity",
    "line_item_value",
    "line_item_insurance",
    "unit_of_measure",
]


# Cap high-cardinality columns to keep feature dimensionality manageable.
HIGH_CARDINALITY_CAP = {
    "vendor": 15,
    "manufacturing_site": 15,
    "country": 20,
    "item_description": 25,
    "molecule_test_type": 20,
    "brand": 20,
    "sub_classification": 20,
    "dosage": 20,
}


def _cap_categories(series: pd.Series, top_n: int) -> pd.Series:
    """Keeps the top_n most frequent categories and buckets the rest as Other."""
    series = series.fillna("Unknown").astype(str)
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other="Other")


def _ensure_downloaded(csv_path: str) -> None:
    """Downloads the real dataset if it is not already available locally."""
    if os.path.exists(csv_path):
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    logger.info(
        "Real dataset not found at %s, downloading from %s",
        csv_path,
        REAL_DATA_URL,
    )

    urllib.request.urlretrieve(REAL_DATA_URL, csv_path)

    logger.info("Downloaded real dataset to %s", csv_path)


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shipment-planning features from date columns."""
    out = df.copy()

    for col in DATE_COLUMN_MAP.values():
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Time between purchase request and purchase order.
    if {"pq_sent_date", "po_sent_date"}.issubset(out.columns):
        out["pq_to_po_days"] = (
            out["po_sent_date"] - out["pq_sent_date"]
        ).dt.days

    # Planned lead time from PO to scheduled delivery.
    if {"po_sent_date", "scheduled_delivery_date"}.issubset(out.columns):
        out["planned_lead_time_days"] = (
            out["scheduled_delivery_date"] - out["po_sent_date"]
        ).dt.days

    # Month and year can capture seasonal and time-related patterns.
    if "po_sent_date" in out.columns:
        out["po_month"] = out["po_sent_date"].dt.month
        out["po_year"] = out["po_sent_date"].dt.year

    # Remove raw datetime columns after feature extraction.
    date_columns = [
        col for col in DATE_COLUMN_MAP.values()
        if col in out.columns
    ]
    out = out.drop(columns=date_columns)

    return out


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add safe engineered predictors.

    None of these features uses the target freight_cost, so they do not directly
    leak the value being predicted.
    """
    out = df.copy()

    if "freight_weight" in out.columns:
        out["log_freight_weight"] = np.log1p(
            out["freight_weight"].clip(lower=0)
        )

    if "line_item_value" in out.columns:
        out["log_line_item_value"] = np.log1p(
            out["line_item_value"].clip(lower=0)
        )

    if {"line_item_value", "line_item_quantity"}.issubset(out.columns):
        quantity = out["line_item_quantity"].replace(0, np.nan)
        out["value_per_item"] = out["line_item_value"] / quantity

    if {"freight_weight", "line_item_quantity"}.issubset(out.columns):
        quantity = out["line_item_quantity"].replace(0, np.nan)
        out["weight_per_item"] = out["freight_weight"] / quantity

    if {"freight_weight", "line_item_value"}.issubset(out.columns):
        out["weight_value_interaction"] = (
            out["freight_weight"] * out["line_item_value"]
        )

    if {"pack_price", "line_item_quantity"}.issubset(out.columns):
        out["pack_price_quantity_interaction"] = (
            out["pack_price"] * out["line_item_quantity"]
        )

    return out


def load_real_data(csv_path: str) -> pd.DataFrame:
    """
    Loads and cleans the SCMS dataset into the format expected by the pipeline.

    Steps:
      1. Load the CSV.
      2. Rename and retain useful numeric/categorical/date columns.
      3. Convert placeholder-laden numeric columns to numeric.
      4. Create date features.
      5. Create engineered numeric features.
      6. Cap high-cardinality categoricals.
      7. Remove invalid target rows.
    """
    _ensure_downloaded(csv_path)

    raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    logger.info(
        "Loaded real dataset: %d rows, %d columns",
        *raw.shape,
    )

    # Rename both normal columns and date columns.
    all_column_map = {
        **RAW_COLUMN_MAP,
        **DATE_COLUMN_MAP,
    }

    df = raw.rename(columns=all_column_map)

    wanted_columns = [
        *RAW_COLUMN_MAP.values(),
        *DATE_COLUMN_MAP.values(),
    ]

    df = df[
        [col for col in wanted_columns if col in df.columns]
    ].copy()

    # Convert numeric columns, including columns containing placeholder strings.
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        before_missing = df[col].isna().mean()
        numeric = pd.to_numeric(df[col], errors="coerce")
        after_missing = numeric.isna().mean()

        non_numeric_rate = after_missing - before_missing

        if non_numeric_rate > 0:
            logger.info(
                "Coercing '%s' to numeric: %.1f%% of rows had "
                "non-numeric placeholder text, now NaN",
                col,
                non_numeric_rate * 100,
            )

        df[col] = numeric

    # Generate date-based features.
    df = _add_date_features(df)

    # Generate additional numeric predictors.
    df = _add_engineered_features(df)

    # Cap high-cardinality categorical columns.
    for col, cap in HIGH_CARDINALITY_CAP.items():
        if col not in df.columns:
            continue

        n_before = df[col].nunique(dropna=True)
        df[col] = _cap_categories(df[col], cap)

        logger.info(
            "Capped '%s': %d levels -> top %d + Other",
            col,
            n_before,
            cap,
        )

    # Drop rows where the prediction target is invalid.
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    # Replace impossible negative date intervals with NaN.
    for col in ["pq_to_po_days", "planned_lead_time_days"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    logger.info(
        "After dropping rows with missing/invalid target: %d rows remain",
        len(df),
    )

    return df.reset_index(drop=True)


def real_continuous_columns() -> List[str]:
    """Continuous columns for IQR outlier treatment on the real dataset."""
    return [
        "freight_weight",
        "pack_price",
        "unit_price",
        "line_item_quantity",
        "line_item_value",
        "line_item_insurance",
        "unit_of_measure",
        "pq_to_po_days",
        "planned_lead_time_days",
        "log_freight_weight",
        "log_line_item_value",
        "value_per_item",
        "weight_per_item",
    ]


if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/SCMS_Delivery_History_Dataset.csv"
    )

    data = load_real_data(path)

    print(data.describe(include="all").T)