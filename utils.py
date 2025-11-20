import os
import numpy as np
import pandas as pd

def safe_float(x, default=np.nan):
    try:
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return default

def sanitize_location_text(location: str):
    """Return (city, state) parsed from 'City, ST'."""
    if not location:
        return "", ""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ""

def load_csv_safe(path: str):
    """Load CSV safely with dtype=str."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return None

def get_safety_data(city: str, state: str):
    """
    New advanced safety model for UrbanIQ.
    Produces safety score, trend, severity level, and realistic rates.
    """
    city = city.lower()

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
        "denver":       {"violent": 5.5, "property": 53.0, "adj": +1}
    }

    default_profile = {"violent": 4.5, "property": 30.0, "adj": 0}
    profile = CITY_CRIME_PROFILE.get(city, default_profile)

    violent_score = max(0, 100 - (profile["violent"] * 7))
    property_score = max(0, 100 - (profile["property"] * 1.2))

    total_safety = int(
        violent_score * 0.55 +
        property_score * 0.40 +
        BASE_SAFE_SCORE * 0.05
    )
    total_safety = max(1, min(95, total_safety + profile["adj"]))

    trend = round((profile["violent"] - 4.0) * 3, 1)

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
        "crime_trend": f"{trend}% YoY",
        "police_response": f"{8 + int(profile['violent'])} min avg",
        "neighborhood_watch": f"{int(total_safety/2)} groups"
    }

def get_quality_data(lat: float, lon: float):
    try:
        distance_from_coast = abs((lon + 100) / 20)
        urban_density = abs(40 - lat) / 10
        base = max(50, min(95, 75 - distance_from_coast + urban_density))

        walkability = int(min(100, base * (1 + urban_density/20)))
        air_quality = int(min(100, base - (distance_from_coast/2)))
        parks = int(min(100, base / 8 + urban_density))
        restaurants = int(min(100, base * (1.5 + urban_density/10)))
        commute = int(max(10, min(90, 35 - base/4 + urban_density)))
        transit = int(min(100, base * (0.8 + urban_density/15)))
        healthcare = int(min(100, base * (0.9 + urban_density/20)))

        return {
            "walkability": f"{walkability}/100",
            "air_quality": f"{air_quality}/100",
            "parks_nearby": parks,
            "restaurants": restaurants,
            "commute_time": f"{commute} min avg",
            "public_transit": f"{transit}/100",
            "healthcare_access": f"{healthcare}/100"
        }
    except Exception:
        return {}

def get_education(city: str):
    return {
        "district_name": f"{city} School District",
        "highest_ranked_school": f"{city} High School",
        "school_rank": "#12",
        "school_rating": "8.2/10",
        "total_schools": "35"
    }

def geocode_city_state(city: str, state: str):
    demo_coords = {
        ("Seattle", "WA"): (47.6062, -122.3321),
        ("Portland", "OR"): (45.5122, -122.6587),
        ("San Francisco", "CA"): (37.7749, -122.4194),
        ("New York", "NY"): (40.7128, -74.0060),
        ("Boston", "MA"): (42.3601, -71.0589),
        ("Chicago", "IL"): (41.8781, -87.6298),
        ("Austin", "TX"): (30.2672, -97.7431),
        ("Denver", "CO"): (39.7392, -104.9903)
    }
    return demo_coords.get((city, state), (None, None))
