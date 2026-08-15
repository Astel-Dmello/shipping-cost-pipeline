"""
Synthetic data generator for the freight shipping-cost pipeline.

Matches the SCHEMA and STATISTICAL PATTERNS described in Jang, Chang & Kim (2023),
"Prediction of Shipping Cost on Freight Brokerage Platform Using Machine Learning",
Sustainability 15(2):1122. The paper's real dataset (1.88M rows, Apr-Sep 2020, a Korean
freight brokerage platform) is private/proprietary (see its Data Availability Statement),
so this generates a smaller synthetic dataset (see rationale below) with the same columns
and, as closely as a synthetic generator can manage, the same correlation structure
between features and target that the paper reports (see 03_synthetic_data_design.yaml).

Row count: 8,000 rather than 1.88M. That's a demo-runtime choice, not a fidelity claim --
enough rows for a stable 80/20 split and 30-fold CV, small enough to run in seconds.
"""
import numpy as np
import pandas as pd

from config import DataConfig
from logging_config import get_logger

logger = get_logger(__name__)


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def generate(config: DataConfig = DataConfig()) -> pd.DataFrame:
    """Generates a synthetic freight dataset matching the schema in 01_data_schema.yaml
    and the target-construction logic in 03_synthetic_data_design.yaml."""
    n_rows, seed = config.n_rows, config.seed
    rng = np.random.default_rng(seed)

    # --- geography ---
    loading_latitude = rng.uniform(33.0, 38.6, n_rows)
    loading_longitude = rng.uniform(125.0, 130.0, n_rows)
    unloading_latitude = rng.uniform(33.0, 38.6, n_rows)
    unloading_longitude = rng.uniform(125.0, 130.0, n_rows)

    linear_distance = _haversine_km(
        loading_latitude, loading_longitude, unloading_latitude, unloading_longitude
    )
    # actual road distance >= straight-line distance; circuity factor ~1.25 on average
    circuity = rng.lognormal(mean=np.log(1.25), sigma=0.12, size=n_rows)
    actual_distance = linear_distance * np.clip(circuity, 1.0, None)

    # --- cargo / vehicle ---
    freight_weight = rng.lognormal(mean=np.log(3000), sigma=0.7, size=n_rows)  # kg, right-skewed
    vehicle_tonnage = rng.choice(
        [1, 2.5, 5, 8, 11, 15, 25], size=n_rows, p=[0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02]
    )
    vehicle_type = rng.choice(
        ["box", "wing_body", "flatbed", "refrigerated", "tanker"],
        size=n_rows, p=[0.32, 0.30, 0.18, 0.12, 0.08],
    )
    type_of_loading = rng.choice(
        ["forklift", "hand", "crane", "none"], size=n_rows, p=[0.4, 0.35, 0.1, 0.15]
    )
    type_of_unloading = rng.choice(
        ["forklift", "hand", "crane", "none"], size=n_rows, p=[0.4, 0.35, 0.1, 0.15]
    )

    # standard_fare: a reference fare, itself already partly a function of distance/tonnage
    standard_fare = (
        50000
        + 400 * linear_distance
        + 8000 * vehicle_tonnage
        + rng.lognormal(mean=0, sigma=0.3, size=n_rows) * 10000
    )

    # --- time ---
    start = pd.Timestamp("2020-04-01")
    end = pd.Timestamp("2020-09-30 23:59:59")
    offsets = rng.uniform(0, (end - start).total_seconds(), n_rows)
    loading_datetime = pd.to_datetime(start) + pd.to_timedelta(offsets, unit="s")
    transit_hours = np.clip(rng.lognormal(mean=np.log(6), sigma=0.6, size=n_rows), 0.5, 96) \
        * (actual_distance / (actual_distance.mean() + 1e-6))
    unloading_datetime = loading_datetime + pd.to_timedelta(transit_hours, unit="h")

    loading_month = loading_datetime.month
    loading_day = loading_datetime.day
    loading_day_of_week = loading_datetime.dayofweek
    loading_time = loading_datetime.hour + loading_datetime.minute / 60.0
    unloading_month = unloading_datetime.month
    unloading_day = unloading_datetime.day
    unloading_day_of_week = unloading_datetime.dayofweek
    unloading_time = unloading_datetime.hour + unloading_datetime.minute / 60.0

    # --- location / company categoricals ---
    provinces = [
        "Seoul", "Gyeonggi", "Incheon", "Gangwon", "Chungbuk", "Chungnam",
        "Jeonbuk", "Jeonnam", "Gyeongbuk", "Gyeongnam", "Busan", "Daegu",
    ]
    prov_p = rng.dirichlet(np.ones(len(provinces)) * 2)
    loading_location = rng.choice(provinces, size=n_rows, p=prov_p)
    unloading_location = rng.choice(provinces, size=n_rows, p=prov_p)

    n_companies = 30
    company_p = np.array([1.0 / (i + 1) for i in range(n_companies)])
    company_p = company_p / company_p.sum()
    company_code = rng.choice([f"C{i:03d}" for i in range(n_companies)], size=n_rows, p=company_p)

    # --- precipitation: intentionally near-zero relationship to target (paper's null finding) ---
    is_summer = np.isin(loading_month, [6, 7, 8])
    rain_p = np.where(is_summer, 0.45, 0.20)
    has_rain = rng.uniform(size=n_rows) < rain_p
    precipitation = np.where(has_rain, rng.lognormal(mean=1.5, sigma=1.0, size=n_rows), 0.0)

    # --- low-signal categorical/id columns (present in schema, not used in target formula) ---
    payment_method = rng.choice(["prepaid", "postpaid", "monthly_settlement"], size=n_rows, p=[0.5, 0.35, 0.15])
    shipping_cost_payment = rng.choice(["shipper_pays", "consignee_pays"], size=n_rows, p=[0.7, 0.3])
    loading_classification = rng.choice(["A", "B", "C"], size=n_rows)
    dispatch_status = rng.choice(["completed", "cancelled", "pending"], size=n_rows, p=[0.9, 0.05, 0.05])
    share_state = rng.choice(["shared", "private"], size=n_rows, p=[0.6, 0.4])
    shipper_number = rng.integers(1000, 9999, size=n_rows).astype(str)
    registrant_key = rng.integers(10000, 99999, size=n_rows).astype(str)
    primary_key = np.arange(n_rows)
    sort_sequence = rng.integers(1, 50, size=n_rows)

    df = pd.DataFrame({
        "primary_key": primary_key,
        "loading_latitude": loading_latitude,
        "loading_longitude": loading_longitude,
        "unloading_latitude": unloading_latitude,
        "unloading_longitude": unloading_longitude,
        "linear_distance": linear_distance,
        "actual_distance": actual_distance,
        "freight_weight": freight_weight,
        "vehicle_tonnage": vehicle_tonnage,
        "vehicle_type": vehicle_type,
        "type_of_loading": type_of_loading,
        "type_of_unloading": type_of_unloading,
        "standard_fare": standard_fare,
        "loading_datetime": loading_datetime,
        "unloading_datetime": unloading_datetime,
        "loading_month": loading_month,
        "loading_day": loading_day,
        "loading_day_of_week": loading_day_of_week,
        "loading_time": loading_time,
        "unloading_month": unloading_month,
        "unloading_day": unloading_day,
        "unloading_day_of_week": unloading_day_of_week,
        "unloading_time": unloading_time,
        "loading_location": loading_location,
        "unloading_location": unloading_location,
        "company_code": company_code,
        "precipitation": precipitation,
        "payment_method": payment_method,
        "shipping_cost_payment": shipping_cost_payment,
        "loading_classification": loading_classification,
        "dispatch_status": dispatch_status,
        "share_state": share_state,
        "shipper_number": shipper_number,
        "registrant_key": registrant_key,
        "sort_sequence": sort_sequence,
    })

    # --- target construction: weighted sum reflecting the paper's reported importance order ---
    # --- target construction: weighted sum reflecting the paper's reported importance
    #     order, PLUS genuine non-linear/interaction/threshold structure. A purely linear
    #     target trivially favors MLR; real freight cost data has non-linear structure
    #     (this is *why* the paper's boosting models beat MLR), so this generator
    #     deliberately includes some, rather than only linear terms:
    #       - a mild convexity in distance (cost doesn't scale perfectly linearly)
    #       - a distance x heavy-vehicle interaction (heavy trucks cost more per km)
    #       - a threshold/step effect for very heavy freight (handling surcharge)
    #     None of these are literally stated in the paper (it doesn't publish its true
    #     cost function -- that's the whole point of the study), but they're a defensible,
    #     documented modeling choice, not an attempt to force a particular result.
    type_unload_effect = df["type_of_unloading"].map(
        {"forklift": 0, "hand": 8000, "crane": 15000, "none": -5000}
    ).values

    heavy_vehicle = (vehicle_tonnage > 8).astype(float)
    heavy_freight_surcharge = np.where(freight_weight > np.quantile(freight_weight, 0.85), 35000, 0)

    base = 40000
    signal = (
        base
        + 850 * linear_distance                        # dominant driver, still ~linear component
        + 0.9 * linear_distance ** 1.15                 # mild convexity: nonlinear in distance
        + 14 * linear_distance * heavy_vehicle           # interaction: heavy trucks cost more per km
        + 120 * actual_distance                          # second distance-related driver
        + 9.0 * freight_weight                           # moderate (paper r ~ 0.27-0.28)
        + heavy_freight_surcharge                         # threshold/step effect (nonlinear)
        + 7000 * vehicle_tonnage                          # moderate (paper r ~ 0.24-0.27)
        + type_unload_effect                              # moderate (paper r ~ 0.23-0.25)
        + 0.045 * standard_fare                           # moderate (paper r ~ 0.20)
        + 0.0 * precipitation                             # paper: no valid contribution -> excluded
    )
    noise = rng.normal(0, signal.std() * 0.15, n_rows)
    shipping_cost = np.clip(signal + noise, 20000, None)
    df["shipping_cost"] = shipping_cost

    # --- inject missingness (so preprocessing has real work to do) ---
    for col in ["freight_weight", "standard_fare", "actual_distance"]:
        miss_idx = rng.choice(n_rows, size=int(n_rows * rng.uniform(0.03, 0.05)), replace=False)
        df.loc[miss_idx, col] = np.nan

    # --- inject outliers (so IQR-based outlier removal has real work to do) ---
    out_idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    mult = rng.uniform(5, 10, size=len(out_idx))
    target_col = rng.choice(["freight_weight", "shipping_cost"], size=len(out_idx))
    for i, col, m in zip(out_idx, target_col, mult):
        df.loc[i, col] = df.loc[i, col] * m

    assert not df.isna().all(axis=None), "generator produced an all-NaN frame"
    assert (df["shipping_cost"] > 0).all(), "shipping_cost must be positive"
    logger.info("Generated %d rows, %d columns", len(df), df.shape[1])
    return df


if __name__ == "__main__":
    data = generate()
    corr = data[["linear_distance", "actual_distance", "freight_weight",
                 "vehicle_tonnage", "standard_fare", "shipping_cost"]].corr()["shipping_cost"]
    logger.info("Correlation with shipping_cost (sanity check vs. paper's Table 2):\n%s",
                 corr.sort_values(ascending=False).to_string())

