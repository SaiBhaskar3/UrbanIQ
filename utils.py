import os
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

def safe_float(x, default=float("nan")):
    """Safely convert a value to float, stripping commas and percent signs."""
    try:
        if pd.isna(x):
            return default
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return default


def sanitize_location_text(location: str) -> Tuple[str, str]:
    """Return (city, state) parsed from 'City, ST'."""
    if not location:
        return "", ""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ""


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV returning DataFrame or None.
    Uses low_memory=False and strips column names.
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, dtype=str, engine="python")
        except Exception:
            return None

    df.columns = [str(c).strip() for c in df.columns]
    return df

def geocode_city_state(city: str, state: str) -> Tuple[Optional[float], Optional[float]]:
    demo_coords = {
        ("Seattle", "WA"): (47.6062, -122.3321),
        ("Portland", "OR"): (45.5122, -122.6587),
        ("San Francisco", "CA"): (37.7749, -122.4194),
        ("New York", "NY"): (40.7128, -74.0060),
        ("Boston", "MA"): (42.3601, -71.0589),
        ("Chicago", "IL"): (41.8781, -87.6298),
        ("Austin", "TX"): (30.2672, -97.7431),
        ("Denver", "CO"): (39.7392, -104.9903),
    }
    return demo_coords.get((city, state), (None, None))

def get_safety_data(city: str, state: str = "") -> dict:
    city_key = (city or "").strip().lower()
    BASE_SAFE_SCORE = 72
    CITY_CRIME_PROFILE = {
        "seattle":      {"violent": 5.1, "property": 55.0, "adj": +3},
        "portland":     {"violent": 5.2, "property": 63.0, "adj": -2},
        "new york":     {"violent": 3.1, "property": 23.0, "adj": +5},
        "san francisco":{"violent": 6.5, "property": 80.0, "adj": -5},
        "los angeles":  {"violent": 5.9, "property": 54.0, "adj": -3},
        "chicago":      {"violent": 9.9, "property": 46.0, "adj": -8},
        "boston":       {"violent": 3.9, "property": 24.0, "adj": +7},
        "austin":       {"violent": 4.3, "property": 36.0, "adj": +4},
        "denver":       {"violent": 5.5, "property": 53.0, "adj": +1},
    }

    default_profile = {"violent": 4.5, "property": 30.0, "adj": 0}
    profile = CITY_CRIME_PROFILE.get(city_key, default_profile)

    violent_score = max(0, 100 - (profile["violent"] * 7))
    property_score = max(0, 100 - (profile["property"] * 1.2))

    total_safety = int(
        violent_score * 0.55 +
        property_score * 0.40 +
        BASE_SAFE_SCORE * 0.05
    )
    total_safety = max(1, min(95, total_safety + profile["adj"]))

    trend = round((profile["violent"] - 4.0) * 3, 1)
    trend_str = (f"+{trend}%" if trend >= 0 else f"{trend}%") + " YoY"

    if total_safety > 80:
        severity = "Very Safe"
    elif total_safety > 70:
        severity = "Safe"
    elif total_safety > 60:
        severity = "Moderately Safe"
    elif total_safety > 50:
        severity = "Some Risk"
    else:
        severity = "High Risk"

    return {
        "crime_index": total_safety,
        "severity": severity,
        "violent_crime_rate": f"{profile['violent']} per 1,000",
        "property_crime_rate": f"{profile['property']} per 1,000",
        "crime_trend": trend_str,
        "police_response": f"{8 + int(profile['violent'])} min avg",
        "neighborhood_watch": f"{int(total_safety/2)} groups",
    }

def get_quality_data(lat: float, lon: float) -> dict:
    try:
        distance_from_coast = abs((lon + 100) / 20)
        urban_density = abs(40 - lat) / 10
        base = max(50, min(95, 75 - distance_from_coast + urban_density))

        walkability = int(min(100, base * (1 + urban_density / 20)))
        air_quality = int(min(100, base - (distance_from_coast / 2)))
        parks = int(min(100, base / 8 + urban_density))
        restaurants = int(min(100, base * (1.5 + urban_density / 10)))
        commute = int(max(10, min(90, 35 - base / 4 + urban_density)))
        transit = int(min(100, base * (0.8 + urban_density / 15)))
        healthcare = int(min(100, base * (0.9 + urban_density / 20)))

        return {
            "walkability": walkability,
            "air_quality": air_quality,
            "parks": parks,
            "restaurants": restaurants,
            "commute_time": f"{commute} min avg",
            "transit": transit,
            "healthcare": healthcare,
        }
    except Exception:
        return {}

def get_education(city: str) -> dict:
    """
    Return synthetic but city-specific education info.
    Different cities get different rank, rating, and total schools.
    """
    if not city:
        city_clean = "Local"
        city_key = ""
    else:
        city_clean = city.strip().title()
        city_key = city.strip().lower()

    CITY_EDU_PROFILE = {
        "seattle":      {"rank": 8,  "rating": 8.7, "total": 52},
        "portland":     {"rank": 15, "rating": 8.1, "total": 48},
        "boston":       {"rank": 5,  "rating": 9.0, "total": 60},
        "new york":     {"rank": 18, "rating": 8.0, "total": 120},
        "san francisco":{"rank": 7,  "rating": 8.8, "total": 40},
        "chicago":      {"rank": 22, "rating": 7.6, "total": 95},
        "austin":       {"rank": 12, "rating": 8.3, "total": 55},
        "denver":       {"rank": 14, "rating": 8.2, "total": 42},
    }

    default_profile = {"rank": 20, "rating": 7.8, "total": 35}

    profile = CITY_EDU_PROFILE.get(city_key, default_profile)

    return {
        "district_name": f"{city_clean} School District",
        "highest_ranked_school": f"{city_clean} High School",
        "school_rank": f"#{profile['rank']}",
        "school_rating": f"{profile['rating']:.1f}/10",
        "total_schools": str(profile["total"]),
    }


def _identify_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of column names that look like month-year (e.g., 'Oct-16' or 'Oct 2016').
    """
    date_cols: List[str] = []

    for c in df.columns:
        name = str(c).strip()
        lower_name = name.lower()

        if lower_name in NON_TIME_COLS:
            continue

        if MONTH_COL_REGEX.match(name):
            date_cols.append(c)
            continue

        parsed = pd.to_datetime("1 " + name, errors="coerce")
        if pd.notna(parsed):
            date_cols.append(c)

    if not date_cols and df.shape[1] > 6:
        date_cols = list(df.columns[6:])

    return date_cols


def _parse_timeseries_from_row(row: pd.Series, date_cols: List[str]) -> pd.Series:
    """Return pd.Series indexed by datetime constructed from date_cols for this row."""
    if not date_cols:
        return pd.Series(dtype=float)

    vals = (
        row.loc[date_cols]
        .replace("", np.nan)
        .map(lambda x: str(x).replace(",", "").strip())
    )

    numeric = pd.to_numeric(vals, errors="coerce")

    parsed = pd.to_datetime(date_cols, errors="coerce", infer_datetime_format=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(
            ["1 " + c for c in date_cols],
            errors="coerce",
            infer_datetime_format=True,
        )

    if parsed.isna().all():
        idx = pd.Index(range(len(date_cols)))
    else:
        idx = parsed

    s = pd.Series(data=numeric.values, index=idx).dropna()
    return s


def get_price_data_for_city(city: str, state: str, df_price: Optional[pd.DataFrame]) -> dict:
    """
    Returns:
      - latest_price: last available non-NaN value (string)
      - median_price: median of the timeseries (string)
      - price_timeseries: pd.Series indexed by datetime (or numeric index) with floats
    """
    if df_price is None or df_price.empty:
        return {"latest_price": "No data", "median_price": "No data", "price_timeseries": None}

    df = df_price.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_cols = _identify_date_columns(df)

    lower_map = {c.lower(): c for c in df.columns}

    city_col = None
    for name in ("city", "city name", "city_name"):
        if name in lower_map:
            city_col = lower_map[name]
            break
    if city_col is None:
        city_candidates = [c for c in df.columns if "city" in c.lower()]
        city_col = city_candidates[0] if city_candidates else None

    state_col = None
    for name in ("state", "st", "state_code"):
        if name in lower_map:
            state_col = lower_map[name]
            break
    if state_col is None:
        state_candidates = [c for c in df.columns if "state" in c.lower()]
        state_col = state_candidates[0] if state_candidates else None

    mask_city = False
    mask_state = False

    try:
        if city_col:
            mask_city = (
                df[city_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == (city or "").strip().lower()
            )
        if state_col:
            mask_state = (
                df[state_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == (state or "").strip().lower()
            )
    except Exception:
        mask_city = False
        mask_state = False

    if isinstance(mask_city, (pd.Series, np.ndarray)) and isinstance(mask_state, (pd.Series, np.ndarray)):
        matches = df[mask_city & mask_state]
    else:
        matches = pd.DataFrame()

    # Fallback: city only
    if matches.empty and isinstance(mask_city, (pd.Series, np.ndarray)):
        matches = df[mask_city]

    if matches.empty and city and city_col:
        try:
            matches = df[
                df[city_col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(city.strip().lower(), na=False)
            ]
        except Exception:
            matches = pd.DataFrame()

    if matches.empty:
        if df.shape[0] == 0:
            return {"latest_price": "No data", "median_price": "No data", "price_timeseries": None}
        row = df.iloc[0]
    else:
        row = matches.iloc[0]

    if date_cols:
        ts = _parse_timeseries_from_row(row, date_cols)
    else:
        ts = pd.Series(dtype=float)

    latest_val = None
    median_val = None

    if not ts.empty:
        ts = ts.sort_index()
        latest_val = ts.iloc[-1]
        median_val = float(ts.median(skipna=True)) if not ts.empty else None
    else:
        latest_val = (
            row.get("Latest Price") or
            row.get("LatestPrice") or
            row.get("Latest_Price") or
            None
        )
        median_val = (
            row.get("Median Price") or
            row.get("MedianPrice") or
            None
        )
        if latest_val is not None:
            try:
                latest_val = float(str(latest_val).replace(",", ""))
            except Exception:
                pass

    def fmt(v):
        return (
            f"{v:.2f}"
            if isinstance(v, (int, float, np.number))
            else (str(v) if v is not None else "No data")
        )

    return {
        "latest_price": fmt(latest_val),
        "median_price": fmt(median_val),
        "price_timeseries": ts,
    }

def get_real_estate_data(city: str, state: str, df_rexus: Optional[pd.DataFrame]) -> dict:
    """
    Returns first matching row from df_rexus or an empty dict if nothing available.
    """
    if df_rexus is None or df_rexus.empty:
        return {}

    df = df_rexus.copy()
    df.columns = [str(c).strip() for c in df.columns]

    city_col = "Bldg City" if "Bldg City" in df.columns else None
    state_col = "Bldg State" if "Bldg State" in df.columns else None

    if city_col is None:
        possible_city = [c for c in df.columns if "city" in c.lower()]
        city_col = possible_city[0] if possible_city else None

    if state_col is None:
        possible_state = [c for c in df.columns if "state" in c.lower()]
        state_col = possible_state[0] if possible_state else None

    mask_city = False
    mask_state = False

    try:
        if city_col:
            mask_city = (
                df[city_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
                == city.strip().upper()
            )
        if state_col:
            mask_state = (
                df[state_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
                == state.strip().upper()
            )
    except Exception:
        mask_city = False
        mask_state = False

    if isinstance(mask_city, (pd.Series, np.ndarray)) and isinstance(mask_state, (pd.Series, np.ndarray)):
        matches = df[mask_city & mask_state]
    else:
        matches = pd.DataFrame()

    if matches.empty and isinstance(mask_city, (pd.Series, np.ndarray)):
        matches = df[mask_city]

    if matches.empty and city_col and city:
        try:
            matches = df[
                df[city_col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(city.strip().upper(), na=False)
            ]
        except Exception:
            matches = pd.DataFrame()

    if matches.empty:
        return df.iloc[0].to_dict()

    return matches.iloc[0].to_dict()

def semantic_retrieve_rexus(
    query: str,
    df: Optional[pd.DataFrame],
    embeddings=None,
    model=None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    If embeddings + model are provided, run cosine similarity search.
    Otherwise, do a simple case-insensitive substring match across all fields.
    """
    if df is None or df.empty or not query:
        return pd.DataFrame()

    if embeddings is not None and model is not None:
        try:
            from numpy.linalg import norm

            q_vec = model.encode([query])[0]
            emb = np.asarray(embeddings)
            sims = emb @ q_vec / (norm(emb, axis=1) * norm(q_vec) + 1e-9)

            top_idx = np.argsort(-sims)[:top_k]
            return df.iloc[top_idx].assign(similarity=sims[top_idx])
        except Exception:
            # If anything goes wrong, fall back to substring
            pass

    mask = df.apply(
        lambda row: row.astype(str).str.contains(query, case=False, na=False).any(),
        axis=1,
    )
    return df[mask].head(top_k)
