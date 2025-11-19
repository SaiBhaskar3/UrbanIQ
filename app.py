import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    sanitize_location_text,
    load_csv_safe,
    init_sentence_model,
    geocode_city_state,
    get_real_estate_data,
    get_safety_data,
    get_quality_data,
    semantic_retrieve_rexus
)

st.set_page_config(
    page_title="UrbanIQ – US Neighborhood Insights",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<style>
.section-title { color:#111827; font-size:1.3rem; font-weight:700; margin: 1rem 0 0.5rem 0; }
.metric-item { background-color:#fff; padding:0.75rem; border-radius:6px; border:1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Data Loading
# -------------------------------
df_rexus = load_csv_safe("data_gov_bldg_rexus.csv")
df_price = load_csv_safe("price.csv")

emb_model = init_sentence_model()
if emb_model and df_rexus is not None:
    combined_col = (
        df_rexus.get("Bldg Address1", "").fillna("") + " " +
        df_rexus.get("Bldg City", "").fillna("") + " " +
        df_rexus.get("Bldg State", "").fillna("")
    )
    rexus_embeddings = emb_model.encode(list(combined_col), show_progress_bar=False)
else:
    rexus_embeddings = None

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("UrbanIQ Navigation")
page = st.sidebar.selectbox("Go to:", ["Compare", "Data Explorer", "Settings"])

with st.sidebar.form("location_form", clear_on_submit=False):
    loc1 = st.text_input("Location A", value="Seattle, WA")
    loc2 = st.text_input("Location B", value="Portland, OR")
    submit_locations = st.form_submit_button("Update")

if "data1" not in st.session_state:
    st.session_state.data1 = None
if "data2" not in st.session_state:
    st.session_state.data2 = None


# -------------------------------
# PAGE 1: Compare
# -------------------------------
if page == "Compare":

    st.header("🏙️ UrbanIQ – Compare Two US Locations")

    left, right = st.columns([1, 3])
    with left:
        st.markdown("### Locations Selected")
        st.write(f"**A:** {loc1}")
        st.write(f"**B:** {loc2}")

        if st.button("Load & Compare"):
            with st.spinner("Fetching neighborhood insights..."):
                city1, state1 = sanitize_location_text(loc1)
                city2, state2 = sanitize_location_text(loc2)

                st.session_state.data1 = {
                    "real_estate": get_real_estate_data(city1, state1, df_rexus),
                    "safety": get_safety_data(city1),
                    "quality": get_quality_data(*geocode_city_state(city1, state1))
                }

                st.session_state.data2 = {
                    "real_estate": get_real_estate_data(city2, state2, df_rexus),
                    "safety": get_safety_data(city2),
                    "quality": get_quality_data(*geocode_city_state(city2, state2))
                }

            st.success("Comparison Ready!")

    # RIGHT SIDE CONTENT
    if st.session_state.data1 and st.session_state.data2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y %I:%M %p')}")

        # -------------------------------
        # Real Estate
        # -------------------------------
        st.subheader("🏠 Real Estate Overview")
        colA, colB = st.columns(2)

        def card(data, title):
            real = data["real_estate"]
            st.markdown(f"### {title}")
            st.markdown("<div class='metric-item'>", unsafe_allow_html=True)
            for k, v in real.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v}")
            st.markdown("</div>", unsafe_allow_html=True)

        with colA:
            card(st.session_state.data1, loc1)

        with colB:
            card(st.session_state.data2, loc2)

        # -------------------------------
        # Safety Bar Chart
        # -------------------------------
        st.subheader("🚓 Safety Comparison")

        safety_df = pd.DataFrame([
            {"Location": loc1, "Score": st.session_state.data1["safety"]["crime_index"]},
            {"Location": loc2, "Score": st.session_state.data2["safety"]["crime_index"]}
        ])

        fig = px.bar(
            safety_df,
            x="Location",
            y="Score",
            text="Score",
            color="Location",
            labels={"Score": "Safety Score (Higher = Safer)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # Quality Radar Chart
        # -------------------------------
        st.subheader("🌿 Quality of Life Radar Chart")

        def quality_df(row, label):
            q = row["quality"]
            return pd.DataFrame({
                "Metric": ["Walkability", "Air Quality", "Transit", "Healthcare", "Restaurants"],
                "Value": [q["walkability"], q["air_quality"], q["transit"], q["healthcare"], q["restaurants"]],
                "Location": label
            })

        radar_data = pd.concat([
            quality_df(st.session_state.data1, loc1),
            quality_df(st.session_state.data2, loc2)
        ])

        fig_radar = px.line_polar(
            radar_data,
            r="Value",
            theta="Metric",
            color="Location",
            line_close=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# -------------------------------
# PAGE 2: Data Explorer
# -------------------------------
elif page == "Data Explorer":

    st.header("🔎 Data Explorer")

    if df_rexus is not None:
        st.subheader("Building Dataset (First 50 Rows)")
        st.dataframe(df_rexus.head(50))

        query = st.text_input("Search Buildings (Semantic)")
        k = st.slider("Top K", 1, 10, 3)

        if st.button("Search"):
            results = semantic_retrieve_rexus(query, k, df_rexus, emb_model, rexus_embeddings)
            if results.empty:
                st.info("No results.")
            else:
                st.dataframe(results)
    else:
        st.info("Upload data_gov_bldg_rexus.csv to enable Data Explorer.")


# -------------------------------
# PAGE 3: Settings
# -------------------------------
elif page == "Settings":

    st.header("⚙️ Settings")
    st.info("UrbanIQ demo mode — add API keys for real data.")

    st.markdown("""
    ### Supported APIs (Optional)
    - Census API  
    - WAQI Air Quality API  
    - Zillow Estimator (if token provided)  
    - FBI Crime Data API  

    Add these to a `.env` file or Streamlit Cloud Secrets.
    """)
