import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import traceback
import os

from utils import (
    sanitize_location_text,
    load_csv_safe,
    geocode_city_state,
    get_safety_data,
    get_quality_data,
    get_education,
    get_price_data_for_city,
    get_real_estate_data,
    semantic_retrieve_rexus,
)

st.set_page_config(
    page_title="UrbanIQ – US Neighborhood Insights",
    page_icon="🏙️",
    layout="wide",
)

st.markdown("<h1 style='text-align:center;'>🏘️ UrbanIQ – US Neighborhood Comparison</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#6b7280;'>Using provided datasets: <code>data_gov_bldg_rexus.csv</code> and <code>price.csv</code></p>", unsafe_allow_html=True)

BASE = os.getcwd()
rexus_path = os.path.join(BASE, "data_gov_bldg_rexus.csv")
price_path = os.path.join(BASE, "price.csv")

df_rexus = load_csv_safe(rexus_path)
df_price = load_csv_safe(price_path)

if df_rexus is None or df_rexus.empty:
    st.warning("REXUS building dataset not found or empty. Upload 'data_gov_bldg_rexus.csv' to the app root.")
if df_price is None or df_price.empty:
    st.warning("Price dataset not found or empty. Upload 'price.csv' to the app root.")

st.sidebar.title("Compare Locations")
loc1 = st.sidebar.text_input("Location A (City, ST)", value="Seattle, WA")
loc2 = st.sidebar.text_input("Location B (City, ST)", value="Portland, OR")
if st.sidebar.button("Compare"):

    try:
        city1, state1 = sanitize_location_text(loc1)
        city2, state2 = sanitize_location_text(loc2)

        lat1, lon1 = geocode_city_state(city1, state1, df_rexus)
        lat2, lon2 = geocode_city_state(city2, state2, df_rexus)

        safety1 = get_safety_data(city1, state1, df_price)
        safety2 = get_safety_data(city2, state2, df_price)

        quality1 = get_quality_data(lat1, lon1, df_price)
        quality2 = get_quality_data(lat2, lon2, df_price)

        edu1 = get_education(city1, state1, df_price, df_rexus)
        edu2 = get_education(city2, state2, df_price, df_rexus)

        price1 = get_price_data_for_city(city1, state1, df_price)
        price2 = get_price_data_for_city(city2, state2, df_price)

        real1 = get_real_estate_data(city1, state1, df_rexus)
        real2 = get_real_estate_data(city2, state2, df_rexus)

        st.session_state["a"] = {"city": city1, "state": state1, "safety": safety1, "quality": quality1, "education": edu1, "price": price1, "real": real1, "coords": (lat1, lon1)}
        st.session_state["b"] = {"city": city2, "state": state2, "safety": safety2, "quality": quality2, "education": edu2, "price": price2, "real": real2, "coords": (lat2, lon2)}

        st.subheader("Overview")
        cols = st.columns(3)
        cols[0].metric(f"{city1}, {state1} Safety", safety1["crime_index"])
        cols[1].metric(f"{city1}, {state1} Quality", quality1.get("composite_score", "N/A"))
        cols[2].metric(f"{city1}, {state1} School Rank", edu1.get("school_rank", "N/A"))

        colsb = st.columns(3)
        colsb[0].metric(f"{city2}, {state2} Safety", safety2["crime_index"])
        colsb[1].metric(f"{city2}, {state2} Quality", quality2.get("composite_score", "N/A"))
        colsb[2].metric(f"{city2}, {state2} School Rank", edu2.get("school_rank", "N/A"))

        st.subheader("City-level Price Time Series (if available)")
        def prepare_ts(data, label):
            if data is None:
                return pd.DataFrame()
            if isinstance(data, pd.Series):
                df = data.to_frame(name=label)
                df.index.name = "Date"
                return df
            if isinstance(data, pd.DataFrame):
                df = data.copy()
                if len(df.columns) > 1:
                    df[label] = df.iloc[:, 0]
                    df = df[[label]]
                else:
                    df.columns = [label]
                df.index.name = "Date"
                return df
            return pd.DataFrame()

        ts1 = prepare_ts(price1.get("price_timeseries"), f"{city1}, {state1}")
        ts2 = prepare_ts(price2.get("price_timeseries"), f"{city2}, {state2}")

        ts_df = pd.concat([t for t in [ts1, ts2] if not t.empty], axis=1) if (not ts1.empty or not ts2.empty) else pd.DataFrame()
        if not ts_df.empty:
            try:
                ts_df.index = pd.to_datetime(ts_df.index, errors="coerce")
                ts_df = ts_df.sort_index()
            except Exception:
                pass
            fig = px.line(ts_df, x=ts_df.index, y=ts_df.columns, markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No city-level timeseries price data available for these cities.")

        st.subheader("Safety Comparison")
        safety_df = pd.DataFrame([
            {"Location": f"{city1}, {state1}", "Score": safety1["crime_index"]},
            {"Location": f"{city2}, {state2}", "Score": safety2["crime_index"]},
        ])
        fig_s = px.bar(safety_df, x="Location", y="Score", text="Score", color="Location")
        st.plotly_chart(fig_s, use_container_width=True)

        st.subheader("Quality of Life Snapshot")
        def make_quality_df(qdict, label):
            return pd.DataFrame({
                "Metric": ["Walkability", "Air Quality", "Transit", "Healthcare", "Restaurants"],
                "Value": [
                    qdict.get("walkability", 0),
                    qdict.get("air_quality", 0),
                    qdict.get("transit", 0),
                    qdict.get("healthcare", 0),
                    qdict.get("restaurants", 0),
                ],
                "Location": label
            })
        qdf = pd.concat([make_quality_df(quality1, f"{city1}, {state1}"), make_quality_df(quality2, f"{city2}, {state2}")])
        fig_q = px.line_polar(qdf, r="Value", theta="Metric", color="Location", line_close=True)
        st.plotly_chart(fig_q, use_container_width=True)

        st.subheader("Education & Real Estate Details")
        left, right = st.columns(2)
        with left:
            st.markdown(f"### {city1}, {state1}")
            st.json(edu1)
            st.write("Real estate sample:")
            st.json(real1)
            st.write("Price summary:")
            st.json(price1)
        with right:
            st.markdown(f"### {city2}, {state2}")
            st.json(edu2)
            st.write("Real estate sample:")
            st.json(real2)
            st.write("Price summary:")
            st.json(price2)

        st.subheader("Building Search (semantic / substring)")
        query = st.text_input("Search buildings (title, address, etc.)")
        k = st.slider("Top K", 1, 10, 5)
        if st.button("Search Buildings"):
            results = semantic_retrieve_rexus(query, df_rexus, embeddings=None, model=None, top_k=k)
            if results is None or results.empty:
                st.info("No results found or dataset/model not available.")
            else:
                st.dataframe(results)

        st.success("Comparison completed.")

    except Exception as e:
        st.error("Error while computing comparison.")
        st.exception(e)
