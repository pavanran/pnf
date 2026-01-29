import os
import glob
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Point & Figure core logic
# -----------------------------
def build_pnf_from_series(prices, box_size=5.0, reversal=3):
    """
    Build Point & Figure columns from a price series (e.g., close or adj_close).
    prices: list/series of floats ordered by time ascending.
    box_size: price increment per box
    reversal: number of boxes for reversal (classic = 3)
    """
    prices = [p for p in prices if pd.notna(p)]
    if len(prices) < 2 or box_size <= 0 or reversal < 1:
        return []

    def quantize(p):
        # Round to nearest box
        return round(p / box_size) * box_size

    last_level = quantize(prices[0])
    columns = []
    current = None  # {"type": "X"/"O", "levels": [...]}

    for i in range(1, len(prices)):
        level = quantize(prices[i])

        # Initialize first column once we have enough movement
        if current is None:
            diff_boxes = int(round((level - last_level) / box_size))
            if abs(diff_boxes) >= reversal:
                if diff_boxes > 0:
                    levels = [last_level + box_size * k for k in range(1, diff_boxes + 1)]
                    current = {"type": "X", "levels": levels}
                else:
                    levels = [last_level - box_size * k for k in range(1, abs(diff_boxes) + 1)]
                    current = {"type": "O", "levels": levels}
                last_level = current["levels"][-1]
            continue

        col_type = current["type"]

        if col_type == "X":
            # Extend up
            if level >= last_level + box_size:
                boxes_up = int(round((level - last_level) / box_size))
                new_levels = [last_level + box_size * k for k in range(1, boxes_up + 1)]
                current["levels"].extend(new_levels)
                last_level = current["levels"][-1]

            # Reverse to O
            elif level <= last_level - (reversal * box_size):
                columns.append(current)
                boxes_down = int(round((last_level - level) / box_size))
                levels = [last_level - box_size * k for k in range(1, boxes_down + 1)]
                current = {"type": "O", "levels": levels}
                last_level = current["levels"][-1]

        else:  # "O"
            # Extend down
            if level <= last_level - box_size:
                boxes_down = int(round((last_level - level) / box_size))
                new_levels = [last_level - box_size * k for k in range(1, boxes_down + 1)]
                current["levels"].extend(new_levels)
                last_level = current["levels"][-1]

            # Reverse to X
            elif level >= last_level + (reversal * box_size):
                columns.append(current)
                boxes_up = int(round((level - last_level) / box_size))
                levels = [last_level + box_size * k for k in range(1, boxes_up + 1)]
                current = {"type": "X", "levels": levels}
                last_level = current["levels"][-1]

    if current is not None:
        columns.append(current)

    return columns


def plot_pnf(columns, title, box_size):
    xs, ys, texts = [], [], []
    for col_idx, col in enumerate(columns):
        for lvl in col["levels"]:
            xs.append(col_idx)
            ys.append(lvl)
            texts.append(col["type"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="text",
            text=texts,
            textfont=dict(size=14),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="P&F Column #",
        yaxis_title=f"Price (box size = {box_size})",
        height=720,
        width=1200,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_yaxes(dtick=box_size)
    fig.update_xaxes(showgrid=False)

    return fig


# -----------------------------
# Data loading (Parquet)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_tickers_from_parquet(folder="."):
    files = glob.glob(os.path.join(folder, "*.parquet"))
    tickers = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
    return tickers


@st.cache_data(show_spinner=False)
def load_ticker_df(ticker, folder="."):
    path = os.path.join(folder, f"{ticker}.parquet")
    df = pd.read_parquet(path)

    # Normalize expected columns
    # Expect columns like: ticker, date, open, high, low, close, adj_close, volume...
    if "date" not in df.columns:
        raise ValueError("Parquet is missing 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# def filter_by_period(df, period_choice, start_date=None, end_date=None):
#     if df.empty:
#         return df

#     max_date = df["date"].max()

#     if period_choice == "Max":
#         return df
#     # Custom range controls
#     start_date = end_date = None
#     if period_choice == "Custom":
#         min_d = df["date"].min().date()
#         max_d = df["date"].max().date()

#         # ✅ Default to last 1 year (or min_d if data is shorter)
#         default_start = (pd.Timestamp(max_d) - pd.DateOffset(years=1)).date()
#         if default_start < min_d:
#             default_start = min_d

#         c1, c2 = st.sidebar.columns(2)
#         start_date = c1.date_input("Start", value=default_start, min_value=min_d, max_value=max_d)
#         end_date = c2.date_input("End", value=max_d, min_value=min_d, max_value=max_d)

#     # if period_choice == "Custom":
#     #     if start_date is None or end_date is None:
#     #         return df
#     #     start = pd.to_datetime(start_date)
#     #     end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
#     #     return df[(df["date"] >= start) & (df["date"] <= end)]
    

#     # rolling windows ending at max_date
#     mapping = {
#         "6M": pd.DateOffset(months=6),
#         "1Y": pd.DateOffset(years=1),
#         "3Y": pd.DateOffset(years=3),
#         "5Y": pd.DateOffset(years=5),
#         "10Y": pd.DateOffset(years=10),
#     }
#     offset = mapping.get(period_choice)
#     if offset is None:
#         return df

#     min_date = max_date - offset
#     return df[df["date"] >= min_date]

def filter_by_period(df, period_choice, start_date=None, end_date=None):
    if df.empty:
        return df

    df = df.copy()
    max_date = df["date"].max()

    # MAX = no filtering
    if period_choice == "Max":
        return df

    # Custom range uses passed start/end
    if period_choice == "Custom":
        if start_date is None or end_date is None:
            return df
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return df[(df["date"] >= start) & (df["date"] <= end)]

    # Rolling window mapping ending at max_date
    mapping = {
        "1D": pd.DateOffset(days=1),
        "5D": pd.DateOffset(days=5),
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5),
        "10Y": pd.DateOffset(years=10),
    }

    offset = mapping.get(period_choice)
    if offset is None:
        return df

    min_date = max_date - offset
    return df[df["date"] >= min_date]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Point & Figure Charts (Parquet)", layout="wide")

st.title("Point & Figure (P&F) Chart Viewer — from Parquet")

folder = st.sidebar.text_input("Data folder (where *.parquet are)", value=".")
tickers = list_tickers_from_parquet(folder)

if not tickers:
    st.error(f"No .parquet files found in folder: {os.path.abspath(folder)}")
    st.stop()

ticker = st.sidebar.selectbox("Ticker", tickers, index=0)

with st.sidebar:
    st.subheader("P&F Settings")
    price_source = st.selectbox("Price source", ["close", "adj_close"], index=0)
    box_size = st.number_input("Box size", min_value=0.01, value=5.0, step=0.5, format="%.2f")
    reversal = st.number_input("Reversal (boxes)", min_value=1, value=3, step=1)

    st.subheader("Period")
    period_choice = st.selectbox("Period", ["1D", "5D", "1M", "3M","6M", "1Y", "3Y", "5Y", "10Y", "Max", "Custom"], index=1)

# Load data
try:
    df = load_ticker_df(ticker, folder)
except Exception as e:
    st.error(f"Failed to load {ticker}.parquet: {e}")
    st.stop()

# Validate price column
if price_source not in df.columns:
    st.error(f"Column '{price_source}' not found in {ticker}.parquet. Available: {list(df.columns)}")
    st.stop()

# Custom range controls
start_date = end_date = None
if period_choice == "Custom":
    min_d = df["date"].min().date()
    max_d = df["date"].max().date()
    c1, c2 = st.sidebar.columns(2)
    # start_date = c1.date_input("Start", value=min_d, min_value=min_d, max_value=max_d)
    # end_date = c2.date_input("End", value=max_d, min_value=min_d, max_value=max_d)
    default_start = (pd.Timestamp(max_d) - pd.DateOffset(years=1)).date()
    if default_start < min_d:
        default_start = min_d

    start_date = c1.date_input("Start", value=default_start, min_value=min_d, max_value=max_d)
    end_date = c2.date_input("End", value=max_d, min_value=min_d, max_value=max_d)


df_f = filter_by_period(df, period_choice, start_date, end_date)

# Build P&F
prices = df_f[price_source].tolist()
cols = build_pnf_from_series(prices, box_size=float(box_size), reversal=int(reversal))

# Header metrics
left, mid, right = st.columns([1.2, 1.2, 2])
with left:
    st.metric("Rows loaded", f"{len(df):,}")
    st.metric("Rows in period", f"{len(df_f):,}")
with mid:
    st.metric("Date range", f"{df_f['date'].min().date()} → {df_f['date'].max().date()}" if not df_f.empty else "—")
    st.metric("P&F columns", f"{len(cols):,}")

with right:
    st.caption("Tip: If you see very few columns, reduce box size or period. If it’s too dense/noisy, increase box size.")

# Plot
title = f"{ticker} Point & Figure ({period_choice}, {price_source}) — box={box_size}, reversal={reversal}"
fig = plot_pnf(cols, title=title, box_size=float(box_size))
st.plotly_chart(fig, use_container_width=True)

# Optional: show raw dataframe
with st.expander("Show raw price data (filtered)"):
    st.dataframe(df_f[["date", "open", "high", "low", "close", "adj_close", "volume"]].tail(300), use_container_width=True)
