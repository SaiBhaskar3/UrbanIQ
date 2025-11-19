import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any
from sentence_transformers import SentenceTransformer


# -----------------------------
# General Utilities
# -----------------------------
def sanitize_location_text(location: str) -> Tuple[str, str]:
    if not location:
        return "", ""
    parts = [p.strip() for p in location.split(",")]
    return (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], "")


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return None


def init_sentence_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# -----------------------------
# Geocoding (Demo Only)
# -----------------------------
def geocode_city_state(city: str, state: str):
    demo_coords = {
        ("Seattle", "WA"): (47.61, -122.33),
        ("Portland", "OR"): (45.52, -122.67),
        ("New York", "NY"): (40.71, -74.00),
        ("Boston", "MA"): (42.36, -71.06),
        ("Chicago", "IL"): (41.88, -87.63),
    }
    return demo_coords.get((city, state), (40.0, -100.0))


# -----------------------------
# Data Modules
# -----------------------------
def get_real_estate_data(city: str, state: str, df):
    if df is None or df.empty:
        return {
            "address": "Demo Address",
            "status": "Active",
            "type": "Office",
            "usable_sqft": "25,000",
            "parking": "50",
            "built": "2010"
        }

    match = df[
        (df["Bldg City"].str.upper() == city.upper()) &
        (df["Bldg State"].str.upper() == state.upper())
    ]

    row = match.iloc[0] if not match.empty else df.iloc[0]

    return {
        "address": row.get("Bldg Address1", "N/A"),
        "status": row.get("Bldg Status", "N/A"),
        "type": row.get("Property Type", "N/A"),
        "usable_sqft": row.get("Bldg ANSI Usable", "N/A"),
        "parking": row.get("Total Parking Spaces", "N/A"),
        "built": row.get("Construction Date", "N/A")
    }


def get_safety_data(city: str):
    safety_base = {
        "Seattle": 78,
        "Boston": 85,
        "Portland": 74,
        "Chicago": 60,
        "New York": 82,
    }
    return {"crime_index": safety_base.get(city, 75)}


def get_quality_data(lat, lon):
    base = max(50, min(95, 80 - abs(lat - 40)))
    return {
        "walkability": int(base * 1.1),
        "air_quality": int(base * 0.9),
        "transit": int(base * 0.85),
        "healthcare": int(base * 1.05),
        "restaurants": int(base * 1.3)
    }


def semantic_retrieve_rexus(query, top_k, df, model, embeddings):
    if df is None or embeddings is None or model is None:
        return pd.DataFrame()
    q_emb = model.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[idx]
