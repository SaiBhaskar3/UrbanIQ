import streamlit as st
import traceback
from datetime import datetime
import pandas as pd
import plotly.express as px

from utils import (
    sanitize_location_text,
    load_csv_safe,
    geocode_city_state,
    get_safety_data,
    get_quality_data,
    get_education,
    get_price_data_for_city,
    get_real_estate_data,
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
.card { background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

df_rexus = load_csv_safe("data_gov_bldg_rexus.csv")
df_price = load_csv_safe("price.csv")

try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

emb_model = None
rexus_embeddings = None
if MODEL_AVAILABLE and df_rexus is not None:
    try:
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        combined_col = (df_rexus.get("Bldg Address1", "").fillna("") + " " +
                        df_rexus.get("Bldg City", "").fillna("") + " " +
                        df_rexus.get("Bldg State", "").fillna(""))
        rexus_embeddings = emb_model.encode(list(combined_col), show_progress_bar=False)
    except Exception:
        emb_model = None
        rexus_embeddings = None

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

st.markdown("<h1 style='text-align:center;'>🏘️ US Neighborhood Comparison (UrbanIQ)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#6b7280;'>Compare two locations across education, housing, safety and quality-of-life.</p>", unsafe_allow_html=True)

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

                # Real estate
                real1 = get_real_estate_data(city1, state1, df_rexus)
                real2 = get_real_estate_data(city2, state2, df_rexus)

                # Safety
                safety1 = get_safety_data(city1, state1)
                safety2 = get_safety_data(city2, state2)

                # Quality
                lat1, lon1 = geocode_city_state(city1, state1)
                lat2, lon2 = geocode_city_state(city2, state2)
                quality1 = get_quality_data(lat1 or 40, lon1 or -100)
                quality2 = get_quality_data(lat2 or 40, lon2 or -100)

                # Education
                edu1 = get_education(city1)
                edu2 = get_education(city2)

                # Price
                price1 = get_price_data_for_city(city1, state1, df_price)
                price2 = get_price_data_for_city(city2, state2, df_price)

                st.session_state.data1 = {
                    "real_estate": real1,
                    "safety": safety1,
                    "quality": quality1,
                    "education": edu1,
                    "price": price1,
                    "coords": {"lat": lat1, "lon": lon1}
                }
                st.session_state.data2 = {
                    "real_estate": real2,
                    "safety": safety2,
                    "quality": quality2,
                    "education": edu2,
                    "price": price2,
                    "coords": {"lat": lat2, "lon": lon2}
                }

            st.success("Comparison Ready!")

    if st.session_state.data1 and st.session_state.data2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y %I:%M %p')}")

        st.subheader("🏠 Real Estate & Price Snapshot")
        colA, colB = st.columns(2)

        def property_card(data, title):
            re = data.get("real_estate", {})
            price = data.get("price", {})
            st.markdown(f"### {title}")
            st.markdown("<div class='metric-item'>", unsafe_allow_html=True)
            if re:
                keys_show = ["Bldg Address1", "Bldg Status", "Property Type", "Bldg ANSI Usable",
                             "Total Parking Spaces", "Owned/Leased", "Construction Date", "Historical Status"]
                for k in keys_show:
                    st.write(f"**{k.replace('_',' ').title()}:** {re.get(k, 'N/A')}")
            else:
                st.write("**No building registry data available.**")
            latest = price.get("latest_price", "No data")
            median = price.get("median_price", "No data")
            st.markdown("---")
            st.write("**Latest Price (city-level):**", latest)
            st.write("**Median Price (city-level):**", median)
            st.markdown("</div>", unsafe_allow_html=True)

        with colA:
            property_card(st.session_state.data1, loc1)
        with colB:
            property_card(st.session_state.data2, loc2)

        p1 = st.session_state.data1["price"].get("price_timeseries")
        p2 = st.session_state.data2["price"].get("price_timeseries")
        ts_df = pd.DataFrame()

        def prepare_ts(data, label):
            if data is None:
                return pd.DataFrame()
            if isinstance(data, pd.Series):
                return data.to_frame(name=label)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
                if len(df.columns) > 1:
                    df[label] = df.iloc[:, 0]
                    return df[[label]]
                else:
                    df.columns = [label]
                    return df
            else:
                return pd.DataFrame()

        ts_df1 = prepare_ts(p1, loc1)
        ts_df2 = 100 

        if not ts_df1.empty and not ts_df2.empty:
            ts_df = pd.concat([ts_df1, ts_df2], axis=1)
        elif not ts_df1.empty:
            ts_df = ts_df1
        elif not ts_df2.empty:
            ts_df = ts_df2

        if not ts_df.empty:
            try:
                parsed_idx = pd.to_datetime(ts_df.index, errors='coerce')
                if parsed_idx.notna().any():
                    ts_df.index = parsed_idx
                ts_df = ts_df.sort_index()
            except Exception:
                pass
            fig_price = px.line(ts_df, x=ts_df.index, y=ts_df.columns, markers=True)
            fig_price.update_layout(xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("No city-level time series price data available for either location.")

        st.subheader("🚓 Safety Comparison")
        safety_df = pd.DataFrame([
            {"Location": loc1, "Score": st.session_state.data1["safety"]["crime_index"]},
            {"Location": loc2, "Score": st.session_state.data2["safety"]["crime_index"]}
        ])
        fig_safety = px.bar(safety_df, x="Location", y="Score", text="Score", color="Location",
                            labels={"Score": "Safety Score (Higher = Safer)"})
        st.plotly_chart(fig_safety, use_container_width=True)

        col1_tr, col2_tr = st.columns(2)
        with col1_tr:
            st.markdown(f"**{loc1} Trend:** {st.session_state.data1['safety']['crime_trend']}")
            st.markdown(f"Severity: {st.session_state.data1['safety']['severity']}")
        with col2_tr:
            st.markdown(f"**{loc2} Trend:** {st.session_state.data2['safety']['crime_trend']}")
            st.markdown(f"Severity: {st.session_state.data2['safety']['severity']}")

        st.subheader("🌿 Quality of Life Radar Chart")

        def quality_df(row, label):
            q = row.get("quality", {})
            def to_num(x):
                if isinstance(x, str) and "/" in x:
                    return float(x.split("/")[0])
                return float(x) if x is not None else 0
            return pd.DataFrame({
                "Metric": ["Walkability", "Air Quality", "Transit", "Healthcare", "Restaurants"],
                "Value": [to_num(q.get("walkability")), to_num(q.get("air_quality")),
                          to_num(q.get("transit")), to_num(q.get("healthcare")), to_num(q.get("restaurants"))],
                "Location": label
            })

        radar_data = pd.concat([quality_df(st.session_state.data1, loc1),
                                quality_df(st.session_state.data2, loc2)])
        fig_radar = px.line_polar(radar_data, r="Value", theta="Metric", color="Location", line_close=True)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("📚 Education & Schools")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            ed = st.session_state.data1.get("education", {})
            st.markdown(f"**{loc1}**")
            for k, v in ed.items():
                st.write(f"{k.replace('_',' ').title()}: {v}")
        with col_e2:
            ed = st.session_state.data2.get("education", {})
            st.markdown(f"**{loc2}**")
            for k, v in ed.items():
                st.write(f"{k.replace('_',' ').title()}: {v}")

elif page == "Data Explorer":
    st.header("🔎 Data Explorer")
    if df_rexus is not None:
        st.subheader("Building Dataset (Sample)")
        st.dataframe(df_rexus.head(50))
        query = st.text_input("Search Buildings (Semantic)")
        k = st.slider("Top K", 1, 10, 3)
        if st.button("Search Buildings"):
            results = semantic_retrieve_rexus(query, df_rexus, rexus_embeddings, emb_model, top_k=k) if MODEL_AVAILABLE else pd.DataFrame()
            if results is None or results.empty:
                st.info("No results or embeddings/model not available.")
            else:
                st.dataframe(results)
    else:
        st.info("Upload 'data_gov_bldg_rexus.csv' in app root to explore building data.")

elif page == "Settings":
    st.header("⚙️ Settings & Diagnostics")
    st.info("UrbanIQ demo mode — add API keys for real data for Census/WAQI/Zillow/FBI")
    st.markdown("### Data files detected:")
    st.write("data_gov_bldg_rexus.csv:", "FOUND" if df_rexus is not None else "NOT FOUND")
    st.write("price.csv:", "FOUND" if df_price is not None else "NOT FOUND")

st.markdown("---")
st.caption("UrbanIQ — Data-driven neighborhood insights. Replace demo data with real CSVs & API keys for production.")
