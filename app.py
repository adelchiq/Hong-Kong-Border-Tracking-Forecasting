"""Streamlit dashboard:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from hkborder import load_raw, load_daily_arrivals, synthetic_raw, walk_forward, summarise

st.set_page_config(page_title="HK Border Traffic Forecast", layout="wide")
st.title("Hong Kong cross-border visitor traffic: analysis and forecasting")
st.caption("Source: Immigration Department, Statistics on Daily Passenger Traffic (data.gov.hk)")


@st.cache_data(show_spinner="Loading data...", ttl=6 * 3600)
def get_data(offline: bool):
    return synthetic_raw() if offline else load_raw()


offline = st.sidebar.toggle("Use synthetic offline data", value=False)
try:
    raw = get_data(offline)
except Exception as e:
    st.error(f"Could not download live data ({e}). Switch on offline data in the sidebar.")
    st.stop()

group = st.sidebar.selectbox("Visitor group", ["mainland_visitors", "other_visitors", "hk_residents"])
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 28, 7)
arr = raw[raw.direction.str.contains("arriv")]
points = sorted(arr.control_point.unique())
chosen = st.sidebar.multiselect("Control points", points, default=points)

y = load_daily_arrivals(arr[arr.control_point.isin(chosen)], group)

tab1, tab2, tab3 = st.tabs(["Overview", "Seasonality", "Forecast backtest"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest 7-day average", f"{y.tail(7).mean():,.0f}")
    prev = y.iloc[-372:-365].mean()
    c2.metric("Same week last year", f"{prev:,.0f}", f"{(y.tail(7).mean() / prev - 1) * 100:+.1f}% YoY")
    c3.metric("Days of data", f"{len(y):,}")
    st.line_chart(pd.DataFrame({"daily": y, "28-day average": y.rolling(28).mean()}))
    by_cp = (arr[arr.date >= y.index.min()].groupby(["date", "control_point"])[group].sum()
             .unstack().rolling(28).mean())
    st.subheader("28-day average by control point")
    st.line_chart(by_cp)

with tab2:
    d = pd.DataFrame({"y": y})
    d["rel"] = d.y / d.y.rolling(28, center=True, min_periods=14).mean()
    d["dow"], d["month"] = d.index.dayofweek, d.index.month
    piv = d.pivot_table(index="dow", columns="month", values="rel", aggfunc="mean")
    piv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.subheader("Arrivals relative to 28-day trend")
    st.dataframe(piv.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"))

with tab3:
    st.write("Walk-forward backtest: every forecast uses only data available "
             f"{horizon} days earlier. Compared against a same-weekday-last-week baseline.")
    if st.button("Run backtest"):
        with st.spinner("Training models..."):
            res = walk_forward(y, horizon=horizon)
        st.dataframe(summarise(res))
        fig, ax = plt.subplots(figsize=(11, 4))
        last = res.iloc[-120:]
        ax.plot(last.index, last.actual, "k", lw=2, label="actual")
        for m in ["naive", "ridge", "gbm"]:
            ax.plot(last.index, last[m], label=m, lw=1)
        ax.legend(); st.pyplot(fig)
