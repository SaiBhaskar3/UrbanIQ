import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

def safe_float(x, default=float("nan")):
    try:
        if pd.isna(x):
            return default
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return default


def sanitize_location_text(location: str) -> Tuple[str, str]:
    """Parse 'City, ST' → ('City', 'ST')"""
    if not location:
        return "", ""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ""


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Safe CSV loader with cleaned columns."""
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

MONTH_COL_REGEX = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s-]?\d{2,4}$",
    re.IGNORECASE
)

NON_TIME_COLS = {
    "city", "city name", "state", "metro", "county", "zipcode", "zip",
    "region", "state code", "st"
}


def _identify_date_columns(df: pd.DataFrame) -> List[str]:
    """Identify month-year columns such as 'Jan-12', 'Feb 2018', etc."""
    date_cols = []

    for col in df.columns:
        name = col.strip()
        lower = name.lower()

        if lower in NON_TIME_COLS:
            continue

        if MONTH_COL_REGEX.match(name):
            date_cols.append(col)
            continue

        try:
            parsed = pd.to_datetime("1 " + name, errors="coerce")
            if pd.notna(parsed):
                date_cols.append(col)
        except Exception:
            pass

    # Fallback heuristic if no obvious time columns but large table
    if not date_cols and df.shape[1] > 6:
        date_cols = list(df.columns[6:])

    return date_cols


def _parse_timeseries_from_row(row: pd.Series, date_cols: List[str]) -> pd.Series:
    """Parse month columns into numeric float timeseries."""
    if not date_cols:
        return pd.Series(dtype=float)

    vals = (
        row.loc[date_cols]
        .replace("", np.nan)
        .map(lambda x: str(x).replace(",", "").strip())
    )
    numeric = pd.to_numeric(vals, errors="coerce")

    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    if parsed_dates.isna().all():
        parsed_dates = pd.to_datetime(["1 " + c for c in date_cols], errors="coerce")

    if parsed_dates.isna().all():
        idx = pd.Index(range(len(date_cols)))
    else:
        idx = parsed_dates

    return pd.Series(numeric.values, index=idx).dropna()

def get_price_data_for_city(city: str, state: str, df: Optional[pd.DataFrame]) -> dict:
    """
    → latest_price: last non-NaN monthly value
    → median_price: median of monthly values
    → price_timeseries: pd.Series
    """
    if df is None or df.empty:
        return {
            "latest_price": "No data",
            "median_price": "No data",
            "price_timeseries": None
        }

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Identify time columns
    date_cols = _identify_date_columns(df)

    # Identify city/state column names dynamically
    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    state_col = next((c for c in df.columns if "state" in c.lower()), None)

    # Match city/state
    filtered = df
    if city_col:
        filtered = filtered[
            filtered[city_col].fillna("").str.lower().str.strip() == city.lower().strip()
        ]

    if state_col and not filtered.empty:
        filtered = filtered[
            filtered[state_col].fillna("").str.lower().str.strip() == state.lower().strip()
        ]

    # Fallback if no match
    if filtered.empty:
        filtered = df[df[city_col].str.contains(city, case=False, na=False)] if city_col else df

    if filtered.empty:
        filtered = df

    row = filtered.iloc[0]

    # Build timeseries
    ts = _parse_timeseries_from_row(row, date_cols)

    if ts.empty:
        return {
            "latest_price": "No data",
            "median_price": "No data",
            "price_timeseries": None
        }

    ts = ts.sort_index()

    latest = ts.iloc[-1]
    median = ts.median()

    return {
        "latest_price": f"{latest:,.0f}",
        "median_price": f"{median:,.0f}",
        "price_timeseries": ts
    }

def get_real_estate_data(city: str, state: str, df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {}

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    state_col = next((c for c in df.columns if "state" in c.lower()), None)

    filtered = df
    if city_col:
        filtered = filtered[
            filtered[city_col].fillna("").str.upper() == city.strip().upper()
        ]
    if state_col and not filtered.empty:
        filtered = filtered[
            filtered[state_col].fillna("").str.upper() == state.strip().upper()
        ]

    if filtered.empty and city_col:
        filtered = df[df[city_col].str.contains(city, case=False, na=False)]

    if filtered.empty:
        return {}

    return filtered.iloc[0].to_dict()


# ------------------------------------------------------------
# School Count + Best Rank
# ------------------------------------------------------------

def get_school_stats(city: str, state: str, df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"school_count": "No data", "best_rank": "No data"}

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    state_col = next((c for c in df.columns if "state" in c.lower()), None)
    rank_col = next((c for c in df.columns if "rank" in c.lower()), None)

    if not city_col or not rank_col:
        return {"school_count": "No data", "best_rank": "No data"}

    filtered = df[df[city_col].fillna("").str.lower() == city.lower().strip()]
    if state_col:
        filtered = filtered[
            filtered[state_col].fillna("").str.lower() == state.lower().strip()
        ]

    if filtered.empty:
        return {"school_count": 0, "best_rank": "No data"}

    school_count = filtered.shape[0]
    best_rank = pd.to_numeric(
        filtered[rank_col].replace("", np.nan), errors="coerce"
    ).min()

    return {
        "school_count": int(school_count),
        "best_rank": int(best_rank) if pd.notna(best_rank) else "No data"
    }

def semantic_retrieve_rexus(query: str, df: Optional[pd.DataFrame],
                            embeddings=None, model=None, top_k: int = 5):
    if df is None or df.empty or not query:
        return pd.DataFrame()

    # If embeddings + model available → real semantic search
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

    # Fallback = substring match
    mask = df.apply(
        lambda row: row.astype(str).str.contains(query, case=False, na=False).any(),
        axis=1,
    )
    return df[mask].head(top_k)
