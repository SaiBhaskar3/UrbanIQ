import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from utils import (
    load_csv_safe,
    sanitize_location_text,
    geocode_city_state,
    get_safety_data,
    get_education,
    get_quality_data,
    safe_float
)

st.set_page_config(page_title="UrbanIQ – City Comparison", layout="wide")

st.sidebar.title("UrbanIQ – Compare Cities")

with st.sidebar.form("loc_form"):
    loc1 = st.text_input("Location A", "Seattle, WA")
    loc2 = st.text_input("Location B", "Portland, OR")
    submit_btn = st.form_submit_button("Compare")

if submit_btn:
    st.session_state["loc1"] = loc1
    st.session_state["loc2"] = loc2

if "loc1" not in st.session_state:
    st.stop()

location1 = st.session_state["loc1"]
location2 = st.session_state["loc2"]

city1, state1 = sanitize_location_text(location1)
city2, state2 = sanitize_location_text(location2)

lat1, lon1 = geocode_city_state(city1, state1)
lat2, lon2 = geocode_city_state(city2, state2)

data1 = {
    "safety": get_safety_data(city1, state1),
    "education": get_education(city1),
    "quality": get_quality_data(lat1 or 40, lon1 or -100)
}

data2 = {
    "safety": get_safety_data(city2, state2),
    "education": get_education(city2),
    "quality": get_quality_data(lat2 or 40, lon2 or -100)
}

st.title("🏙️ UrbanIQ — Neighborhood Intelligence Engine")
st.write(f"Updated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")

st.subheader("🚓 Crime & Safety Comparison")

crime_df = pd.DataFrame([
    {"Location": location1, "Score": data1["safety"]["crime_index"]},
    {"Location": location2, "Score": data2["safety"]["crime_index"]},
])

fig = px.bar(
    crime_df,
    x="Location",
    y="Score",
    text="Score",
    color="Location",
    title="Safety Score (Higher = Safer)"
)
st.plotly_chart(fig, use_container_width=True)

colA, colB = st.columns(2)

with colA:
    st.markdown(f"### {location1}")
    st.write("Severity:", data1["safety"]["severity"])
    st.write("Violent Crime:", data1["safety"]["violent_crime_rate"])
    st.write("Property Crime:", data1["safety"]["property_crime_rate"])
    st.write("Trend:", data1["safety"]["crime_trend"])

with colB:
    st.markdown(f"### {location2}")
    st.write("Severity:", data2["safety"]["severity"])
    st.write("Violent Crime:", data2["safety"]["violent_crime_rate"])
    st.write("Property Crime:", data2["safety"]["property_crime_rate"])
    st.write("Trend:", data2["safety"]["crime_trend"])

st.subheader("✨ Quality of Life Comparison")

def radar_df(data, label):
    q = data["quality"]
    vals = {
        "Walkability": safe_float(q["walkability"]),
        "Air Quality": safe_float(q["air_quality"]),
        "Transit": safe_float(q["public_transit"]),
        "Healthcare": safe_float(q["healthcare_access"]),
        "Restaurants": safe_float(q["restaurants"]),
    }
    df = pd.DataFrame([vals])
    df["label"] = label
    return df

r1 = radar_df(data1, location1)
r2 = radar_df(data2, location2)
df_radar = pd.concat([r1, r2], ignore_index=True)
melted = df_radar.melt(id_vars="label", var_name="metric", value_name="value")

fig2 = px.line_polar(melted, theta="metric", r="value", color="label", line_close=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📚 Education Overview")

colE1, colE2 = st.columns(2)

with colE1:
    st.markdown(f"### {location1}")
    for k, v in data1["education"].items():
        st.write(f"**{k.replace('_',' ').title()}:** {v}")

with colE2:
    st.markdown(f"### {location2}")
    for k, v in data2["education"].items():
        st.write(f"**{k.replace('_',' ').title()}:** {v}")

st.markdown("---")
st.write("Built with ❤️ by UrbanIQ — Data-Driven City Intelligence")
