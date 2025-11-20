import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import traceback

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

st.markdown(
    """
<style>
.section-title { color:#111827; font-size:1.3rem; font-weight:700; margin: 1rem 0 0.5rem 0; }
.metric-item { background-color:#fff; padding:0.75rem; border-radius:6px; border:1px solid #e5e7eb; }
.card { background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_rexus():
    return load_csv_safe("data_gov_bldg_rexus.csv")

@st.cache_data(show_spinner=False)
def load_price():
    return load_csv_safe("price.csv")

df_rexus = load_rexus()
df_price = load_price()


try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not MODEL_AVAILABLE:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def build_rexus_embeddings():
    """
    Build embeddings for the REXUS dataset using the cached model and data.
    No unhashable arguments are passed to keep Streamlit caching happy.
    """
    df = load_rexus()
    model = load_embedding_model()

    if model is None or df is None or df.empty:
        return None

    try:
        addr = df.get("Bldg Address1", "").fillna("")
        city = df.get("Bldg City", "").fillna("")
        state = df.get("Bldg State", "").fillna("")

        combined = (addr + " " + city + " " + state).astype(str).tolist()
        return model.encode(combined, show_progress_bar=False)
    except Exception:
        return None

emb_model = load_embedding_model()
rexus_embeddings = build_rexus_embeddings()

st.sidebar.title("UrbanIQ Navigation")
page = st.sidebar.selectbox("Go to:", ["Compare", "Data Explorer"])

with st.sidebar.form("location_form", clear_on_submit=False):
    loc1 = st.text_input("Location A", value="Seattle, WA")
    loc2 = st.text_input("Location B", value="Portland, OR")
    submit_locations = st.form_submit_button("Update")

# Initialize session state
if "data1" not in st.session_state:
    st.session_state.data1 = None
if "data2" not in st.session_state:
    st.session_state.data2 = None

st.markdown(
    "<h1 style='text-align:center;'>🏘️ US Neighborhood Comparison (UrbanIQ)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>Compare two locations across education, housing, safety and quality-of-life.</p>",
    unsafe_allow_html=True,
)

if page == "Compare":
    st.header("🏙️ UrbanIQ – Compare Two US Locations")
    left, right = st.columns([1, 3])

    with left:
        st.markdown("### Locations Selected")
        st.write(f"**A:** {loc1}")
        st.write(f"**B:** {loc2}")

        if st.button("Load & Compare"):
            try:
                with st.spinner("Fetching neighborhood insights..."):
                    city1, state1 = sanitize_location_text(loc1)
                    city2, state2 = sanitize_location_text(loc2)

                    # Real estate
                    real1 = get_real_estate_data(city1, state1, df_rexus)
                    real2 = get_real_estate_data(city2, state2, df_rexus)

                    # Safety
                    safety1 = get_safety_data(city1, state1)
                    safety2 = get_safety_data(city2, state2)

                    # Quality (fallback coords if geocode missing)
                    lat1, lon1 = geocode_city_state(city1, state1)
                    lat2, lon2 = geocode_city_state(city2, state2)
                    quality1 = get_quality_data(lat1 or 40.0, lon1 or -100.0)
                    quality2 = get_quality_data(lat2 or 40.0, lon2 or -100.0)

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
                        "coords": {"lat": lat1, "lon": lon1},
                    }
                    st.session_state.data2 = {
                        "real_estate": real2,
                        "safety": safety2,
                        "quality": quality2,
                        "education": edu2,
                        "price": price2,
                        "coords": {"lat": lat2, "lon": lon2},
                    }

                st.success("Comparison Ready!")
            except Exception as e:
                st.error("Something went wrong while loading comparison data.")
                st.exception(e)

    if st.session_state.data1 and st.session_state.data2:
        st.markdown(
            f"**Last Updated:** {datetime.now().strftime('%B %d, %Y %I:%M %p')}"
        )

        st.subheader("🏠 Real Estate & Price Snapshot")
        colA, colB = st.columns(2)

        def property_card(data, title):
            re_info = data.get("real_estate", {}) or {}
            price_info = data.get("price", {}) or {}

            st.markdown(f"### {title}")
            st.markdown("<div class='metric-item'>", unsafe_allow_html=True)

            if re_info:
                keys_show = [
                    "Bldg Address1",
                    "Bldg Status",
                    "Property Type",
                    "Bldg ANSI Usable",
                    "Total Parking Spaces",
                    "Owned/Leased",
                    "Construction Date",
                    "Historical Status",
                ]
                for k in keys_show:
                    if k in re_info:
                        st.write(f"**{k.replace('_',' ').title()}:** {re_info.get(k, 'N/A')}")
            else:
                st.write("**No building registry data available.**")

            latest = price_info.get("latest_price", "No data")
            median = price_info.get("median_price", "No data")
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

        def prepare_ts(data, label):
            if data is None or isinstance(data, (float, int)) and pd.isna(data):
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

        ts_df1 = prepare_ts(p1, loc1)
        ts_df2 = prepare_ts(p2, loc2)

        if not ts_df1.empty and not ts_df2.empty:
            ts_df = pd.concat([ts_df1, ts_df2], axis=1)
        elif not ts_df1.empty:
            ts_df = ts_df1
        elif not ts_df2.empty:
            ts_df = ts_df2
        else:
            ts_df = pd.DataFrame()

        if not ts_df.empty:
            try:
                idx_parsed = pd.to_datetime(ts_df.index, errors="coerce")
                if idx_parsed.notna().any():
                    ts_df = ts_df.set_index(idx_parsed)
                ts_df = ts_df.sort_index()
            except Exception:
                pass

            fig_price = px.line(ts_df, x=ts_df.index, y=ts_df.columns, markers=True)
            fig_price.update_layout(xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info(
                "No city-level time series price data available for either location."
            )

        st.subheader("🚓 Safety Comparison")
        safety_df = pd.DataFrame(
            [
                {
                    "Location": loc1,
                    "Score": st.session_state.data1["safety"]["crime_index"],
                },
                {
                    "Location": loc2,
                    "Score": st.session_state.data2["safety"]["crime_index"],
                },
            ]
        )

        fig_safety = px.bar(
            safety_df,
            x="Location",
            y="Score",
            text="Score",
            color="Location",
            labels={"Score": "Safety Score (Higher = Safer)"},
        )
        st.plotly_chart(fig_safety, use_container_width=True)

        col1_tr, col2_tr = st.columns(2)
        with col1_tr:
            s1 = st.session_state.data1["safety"]
            st.markdown(f"**{loc1} Trend:** {s1['crime_trend']}")
            st.markdown(f"Severity: {s1['severity']}")
        with col2_tr:
            s2 = st.session_state.data2["safety"]
            st.markdown(f"**{loc2} Trend:** {s2['crime_trend']}")
            st.markdown(f"Severity: {s2['severity']}")

        st.subheader("🌿 Quality of Life Radar Chart")

        def quality_df(row, label):
            q = row.get("quality", {}) or {}

            def to_num(x):
                if isinstance(x, str) and "/" in x:
                    try:
                        return float(x.split("/")[0])
                    except Exception:
                        return 0.0
                try:
                    return float(x)
                except Exception:
                    return 0.0

            return pd.DataFrame(
                {
                    "Metric": [
                        "Walkability",
                        "Air Quality",
                        "Transit",
                        "Healthcare",
                        "Restaurants",
                    ],
                    "Value": [
                        to_num(q.get("walkability")),
                        to_num(q.get("air_quality")),
                        to_num(q.get("transit")),
                        to_num(q.get("healthcare")),
                        to_num(q.get("restaurants")),
                    ],
                    "Location": label,
                }
            )

        radar_data = pd.concat(
            [quality_df(st.session_state.data1, loc1), quality_df(st.session_state.data2, loc2)]
        )
        fig_radar = px.line_polar(
            radar_data,
            r="Value",
            theta="Metric",
            color="Location",
            line_close=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("📚 Education & Schools")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            ed = st.session_state.data1.get("education", {}) or {}
            st.markdown(f"**{loc1}**")
            for k, v in ed.items():
                st.write(f"{k.replace('_',' ').title()}: {v}")
        with col_e2:
            ed = st.session_state.data2.get("education", {}) or {}
            st.markdown(f"**{loc2}**")
            for k, v in ed.items():
                st.write(f"{k.replace('_',' ').title()}: {v}")

elif page == "Data Explorer":
    st.header("🔎 Data Explorer")

    if df_rexus is not None and not df_rexus.empty:
        st.subheader("Building Dataset (Sample)")
        st.dataframe(df_rexus.head(50))

        query = st.text_input("Search Buildings (Semantic / Text)")
        k = st.slider("Top K", 1, 10, 3)

        if st.button("Search Buildings"):
            try:
                results = semantic_retrieve_rexus(
                    query, df_rexus, rexus_embeddings, emb_model, top_k=k
                )
                if results is None or results.empty:
                    st.info("No results (or embeddings/model not available).")
                else:
                    st.dataframe(results)
            except Exception as e:
                st.error("Error while searching buildings.")
                st.exception(e)
    else:
        st.info(
            "Upload 'data_gov_bldg_rexus.csv' in app root to explore building data."
        )
