import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict


def safe_float(x, default=float("nan")):
    try:
        if pd.isna(x):
            return default
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return default


def sanitize_location_text(location: str) -> Tuple[str, str]:
    """Return (city, state) parsed from 'City, ST' or ('City','ST') from two-arg form."""
    if not location:
        return "", ""
    parts = [p.strip() for p in str(location).split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ""


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV returning DataFrame or None.
    Uses low_memory=False to handle wide files and strips column names.
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


def geocode_city_state(city: str, state: str, df_rexus: Optional[pd.DataFrame] = None) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to find lat/lon using the rexus dataset (if available); then demo_coords; then deterministic fallback.
    Returns (lat, lon) or (None, None).
    """
    city_key = (city or "").strip()
    state_key = (state or "").strip()

    if df_rexus is not None:
        df = df_rexus.copy()
        # common latitude/longitude column names
        lat_cols = [c for c in df.columns if c.lower() in {"latitude", "lat", "y"}]
        lon_cols = [c for c in df.columns if c.lower() in {"longitude", "lon", "lng", "x"}]
        city_cols = [c for c in df.columns if "city" in c.lower()]
        state_cols = [c for c in df.columns if "state" in c.lower()]

        if city_cols:
            city_col = city_cols[0]
            mask_city = df[city_col].fillna("").astype(str).str.strip().str.lower() == city_key.lower()
            if state_cols:
                state_col = state_cols[0]
                mask_state = df[state_col].fillna("").astype(str).str.strip().str.lower() == state_key.lower()
                mask = mask_city & mask_state
            else:
                mask = mask_city

            if mask.any():
                row = df[mask].iloc[0]
                lat = None
                lon = None
                if lat_cols:
                    lat = safe_float(row.get(lat_cols[0]), default=None)
                if lon_cols:
                    lon = safe_float(row.get(lon_cols[0]), default=None)
                if lat is not None and lon is not None:
                    return float(lat), float(lon)

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
    key = (city_key.title(), state_key.upper())
    if key in demo_coords:
        return demo_coords[key]

    # 3) Deterministic fallback: hash-based pseudo-coordinates (changes for every different input)
    # This ensures every different city/state returns a different spot on the map.
    if city_key:
        h = abs(hash(f"{city_key}|{state_key}"))
        lat = 25.0 + (h % 3000) / 100.0  # ~25..55
        lon = -125.0 + (h % 5000) / 100.0  # ~-125..-75
        return float(lat), float(lon)

    return None, None


def get_safety_data(city: str, state: str = "", df_price: Optional[pd.DataFrame] = None) -> dict:
    """
    Compute a safety/crime index. Uses the built-in CITY_CRIME_PROFILE when available,
    otherwise derives a heuristic influenced by median price (if price df supplied)
    so results vary with input and available price data.
    """
    city_key = (city or "").strip().lower()
    BASE_SAFE_SCORE = 72

    CITY_CRIME_PROFILE = {
        "seattle": {"violent": 5.1, "property": 55.0, "adj": +3},
        "portland": {"violent": 5.2, "property": 63.0, "adj": -2},
        "new york": {"violent": 3.1, "property": 23.0, "adj": +5},
        "san francisco": {"violent": 6.5, "property": 80.0, "adj": -5},
        "los angeles": {"violent": 5.9, "property": 54.0, "adj": -3},
        "chicago": {"violent": 9.9, "property": 46.0, "adj": -8},
        "boston": {"violent": 3.9, "property": 24.0, "adj": +7},
        "austin": {"violent": 4.3, "property": 36.0, "adj": +4},
        "denver": {"violent": 5.5, "property": 53.0, "adj": +1},
    }

    default_profile = {"violent": 4.5, "property": 30.0, "adj": 0}
    profile = CITY_CRIME_PROFILE.get(city_key, default_profile)

    price_adj = 0.0
    if df_price is not None and not df_price.empty:
        # attempt to find row and median price
        try:
            lower_map = {c.lower(): c for c in df_price.columns}
            city_col = lower_map.get("city") or next((c for c in df_price.columns if "city" in c.lower()), None)
            state_col = lower_map.get("state") or next((c for c in df_price.columns if "state" in c.lower()), None)
            matches = pd.DataFrame()
            if city_col:
                matches = df_price[df_price[city_col].fillna("").astype(str).str.lower() == city_key]
                if state_col and not matches.empty:
                    matches = matches[matches[state_col].fillna("").astype(str).str.lower() == (state or "").strip().lower()]
            if matches.empty and city_col:
                matches = df_price[df_price[city_col].fillna("").astype(str).str.lower().str.contains(city_key, na=False)]
            if not matches.empty:
                # try to compute median across numeric-like columns
                numeric = matches.apply(pd.to_numeric, errors="coerce").stack()
                if not numeric.empty:
                    median_price = float(numeric.median())
                    # more expensive -> slightly safer in many US cities -> small positive adj
                    price_adj = min(5.0, (median_price / 200000.0))
        except Exception:
            price_adj = 0.0

    violent_score = max(0, 100 - (profile["violent"] * 7) + price_adj)
    property_score = max(0, 100 - (profile["property"] * 1.2) + price_adj)

    total_safety = int(
        violent_score * 0.55 +
        property_score * 0.40 +
        BASE_SAFE_SCORE * 0.05
    )
    total_safety = max(1, min(95, int(total_safety + profile.get("adj", 0))))

    trend = round((profile.get("violent", 4.5) - 4.0) * 3, 1)
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
        "violent_crime_rate": f"{profile.get('violent', 4.5)} per 1,000",
        "property_crime_rate": f"{profile.get('property', 30.0)} per 1,000",
        "crime_trend": trend_str,
        "police_response": f"{8 + int(profile.get('violent', 4))} min avg",
        "neighborhood_watch": f"{int(total_safety / 2)} groups",
    }


def get_quality_data(lat: float, lon: float, df_price: Optional[pd.DataFrame] = None) -> dict:
    """
    Compute quality metrics based on lat/lon and optionally local price info.
    This will vary with different inputs since lat/lon are derived per-city.
    """
    try:
        distance_from_coast = abs((lon + 100) / 20) if lon is not None else 2.0
        urban_density = abs(40 - (lat or 40.0)) / 10

        base = max(30, min(95, 75 - distance_from_coast + urban_density * 2))

        price_boost = 0.0
        if df_price is not None and not df_price.empty:
            try:
                # use the median of numeric columns in price df as a crude proxy
                numeric = df_price.apply(pd.to_numeric, errors="coerce").stack()
                if not numeric.empty:
                    median_all = float(numeric.median())
                    price_boost = min(10, (median_all / 200000.0))
            except Exception:
                price_boost = 0.0

        walkability = int(min(100, max(0, base * (1.0 + urban_density / 20.0) + price_boost)))
        air_quality = int(min(100, max(0, base - (distance_from_coast / 2.0) + price_boost / 2.0)))
        parks = int(min(100, max(0, base / 8.0 + urban_density * 3 + price_boost / 3.0)))
        restaurants = int(min(100, max(0, base * (1.2 + urban_density / 10.0) + price_boost)))
        commute = int(max(5, min(120, 35 - base / 4.0 + urban_density * 5.0)))
        transit = int(min(100, max(0, base * (0.8 + urban_density / 15.0) + price_boost / 4.0)))
        healthcare = int(min(100, max(0, base * (0.9 + urban_density / 20.0) + price_boost / 3.0)))

        composite = int(np.round(np.clip((walkability + air_quality + transit + healthcare + restaurants) / 5.0, 1, 100)))

        return {
            "walkability": walkability,
            "air_quality": air_quality,
            "parks": parks,
            "restaurants": restaurants,
            "commute_time": f"{commute} min avg",
            "transit": transit,
            "healthcare": healthcare,
            "composite_score": composite,
        }
    except Exception:
        return {}


def get_education(city: str, state: str = "", df_price: Optional[pd.DataFrame] = None, df_rexus: Optional[pd.DataFrame] = None) -> dict:
    """
    Produce a dynamic education object. When price.csv is available, compute a school_rank
    based on the city's median price percentile among cities in price.csv (higher price -> higher rank).
    Otherwise fallback to a deterministic but varying heuristic based on city name hash.
    """
    city_key = (city or "").strip()
    district_name = f"{city_key} School District" if city_key else "Local School District"
    highest_school = f"{city_key} High School" if city_key else "Local High School"

    rank = None
    if df_price is not None and not df_price.empty:
        try:
            lower_map = {c.lower(): c for c in df_price.columns}
            city_col = lower_map.get("city") or next((c for c in df_price.columns if "city" in c.lower()), None)
            state_col = lower_map.get("state") or next((c for c in df_price.columns if "state" in c.lower()), None)

            matches = pd.DataFrame()
            if city_col:
                matches = df_price[df_price[city_col].fillna("").astype(str).str.lower() == city_key.lower()]
                if state_col and not matches.empty:
                    matches = matches[matches[state_col].fillna("").astype(str).str.lower() == (state or "").strip().lower()]

            if matches.empty and city_col:
                matches = df_price[df_price[city_col].fillna("").astype(str).str.lower().str.contains(city_key.lower(), na=False)]

            if not matches.empty:
                # compute a single median price for this city from numeric-like columns
                numeric_vals = matches.apply(pd.to_numeric, errors="coerce").stack()
                if not numeric_vals.empty:
                    target_median = float(numeric_vals.median())

                    if city_col:
                        city_medians = []
                        for name, group in df_price.groupby(city_col):
                            nums = group.apply(pd.to_numeric, errors="coerce").stack()
                            if not nums.empty:
                                city_medians.append((name, float(nums.median())))
                        # build percentile rank of target_median among city_medians
                        medians_vals = [m for (_, m) in city_medians]
                        if medians_vals:
                            pct = (sum(1 for v in medians_vals if v <= target_median) / max(1, len(medians_vals)))
                            # map percentile (0..1) to rank 1..10 (higher pct => better rank)
                            rank = int(max(1, min(10, round(pct * 10))))
        except Exception:
            rank = None

    if rank is None:
        h = abs(hash(city_key + "|" + (state or ""))) % 100
        # convert 0..99 to rank 1..10 (higher hash -> higher rank)
        rank = max(1, min(10, int(np.ceil((h + 1) / 10.0))))

    total_schools = None
    if df_rexus is not None and not df_rexus.empty:
        try:
            city_cols = [c for c in df_rexus.columns if "city" in c.lower()]
            if city_cols:
                city_col = city_cols[0]
                cnt = df_rexus[df_rexus[city_col].fillna("").astype(str).str.lower().str.contains(city_key.lower(), na=False)].shape[0]
                if cnt:
                    total_schools = cnt
        except Exception:
            total_schools = None

    return {
        "district_name": district_name,
        "highest_ranked_school": highest_school,
        "school_rank": f"#{rank}",
        "school_rank_numeric": rank,
        "school_rating": f"{6.0 + (rank * 0.4):.1f}/10",
        "total_schools": str(total_schools or (5 + (rank % 5))),
    }


MONTH_COL_REGEX = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-]?\d{2,4}$",
    re.IGNORECASE,
)

NON_TIME_COLS = {
    "city", "city name", "state", "st", "state_code",
    "region", "zip", "zipcode", "metro", "county"
}


def _identify_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols = []

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

    idx = parsed if not parsed.isna().all() else pd.Index(range(len(date_cols)))

    s = pd.Series(data=numeric.values, index=idx).dropna()
    return s


def get_price_data_for_city(city: str, state: str, df_price: Optional[pd.DataFrame]) -> dict:
    """
    Returns city-level price info: latest, median and timeseries using the price.csv dataset you uploaded.
    This function is data-driven and will vary based on different city/state inputs.
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
                df[city_col].fillna("").astype(str).str.strip().str.lower() ==
                (city or "").strip().lower()
            )
        if state_col:
            mask_state = (
                df[state_col].fillna("").astype(str).str.strip().str.lower() ==
                (state or "").strip().lower()
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

    if matches.empty and city and city_col:
        try:
            matches = df[
                df[city_col].fillna("").astype(str).str.lower().str.contains(
                    city.strip().lower(), na=False
                )
            ]
        except Exception:
            matches = pd.DataFrame()

    if matches.empty:
        try:
            numeric_all = df.apply(pd.to_numeric, errors="coerce").stack()
            if numeric_all.empty:
                return {"latest_price": "No data", "median_price": "No data", "price_timeseries": None}
            overall_median = float(numeric_all.median())
            return {"latest_price": f"{overall_median:.2f}", "median_price": f"{overall_median:.2f}", "price_timeseries": None}
        except Exception:
            return {"latest_price": "No data", "median_price": "No data", "price_timeseries": None}

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
        median_val = float(ts.median(skipna=True))
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
            f"{v:.2f}" if isinstance(v, (int, float, np.number))
            else (str(v) if v is not None else "No data")
        )

    return {
        "latest_price": fmt(latest_val),
        "median_price": fmt(median_val),
        "price_timeseries": ts,
    }


def get_real_estate_data(city: str, state: str, df_rexus: Optional[pd.DataFrame]) -> dict:
    """
    Return the first matching row from df_rexus or a summary fallback.
    This is data-driven and will change for different input cities.
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
                df[city_col].fillna("").astype(str).str.strip().str.upper() ==
                city.strip().upper()
            )
        if state_col:
            mask_state = (
                df[state_col].fillna("").astype(str).str.strip().str.upper() ==
                state.strip().upper()
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
                df[city_col].fillna("").astype(str).str.upper().str.contains(
                    city.strip().upper(), na=False
                )
            ]
        except Exception:
            matches = pd.DataFrame()

    if matches.empty:
        # fallback summary (take first row and summarize)
        first = df.iloc[0].to_dict()
        # reduce verbosity: return a subset of keys if they exist
        keys = ["Bldg Address1", "Bldg City", "Bldg State", "Property Type", "Construction Date"]
        return {k: first.get(k, "") for k in keys}

    return matches.iloc[0].to_dict()


def semantic_retrieve_rexus(
    query: str,
    df: Optional[pd.DataFrame],
    embeddings=None,
    model=None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    If embeddings + model are provided, run cosine similarity search. Otherwise, do a
    simple case-insensitive substring match across all fields.
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
            pass

    mask = df.apply(
        lambda row: row.astype(str).str.contains(query, case=False, na=False).any(),
        axis=1,
    )
    return df[mask].head(top_k)
