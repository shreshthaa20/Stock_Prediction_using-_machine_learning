import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px

import config as cfg
from data_loader import load_ticker, make_sequences, prepare_data
from evaluate import metrics


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StockSight",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Load custom CSS ────────────────────────────────────────────────────────

def _load_css():
    css_path = os.path.join("assets", "custom.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()




# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "Actual": "#6B7280",
    "ANN": "#059669",
    "RFR": "#2563EB",
    "LSTM": "#7C3AED",
}
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI"]


# ─── Plotly helper functions & Interactive Controls ─────────────────────────

PLOTLY_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "drawline", "drawopenpath", "eraseshape", "resetScale2d",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "stocksight_chart",
        "height": 500,
        "width": 1000,
        "scale": 2,
    },
    "responsive": True,
}
COMPARISON_PLOTLY_CONFIG = {
    **PLOTLY_CONFIG,
    "modeBarButtonsToRemove": [
        *PLOTLY_CONFIG["modeBarButtonsToRemove"],
        "select2d", "lasso2d", "autoScale2d",
    ],
}


def plotly_prediction_chart(x_axis, actual, predictions_dict, title, show_flags=None):
    """
    Interactive Plotly line chart: Actual vs one or more model predictions.
    Displays actual values alongside one or more model predictions.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(color=MODEL_COLORS["Actual"], dash="dash", width=2),
    ))
    for name, pred in predictions_dict.items():
        if show_flags and not show_flags.get(name, True):
            continue
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=pred,
            mode="lines",
            name=f"{name} Predicted",
            line=dict(color=MODEL_COLORS.get(name, "#FFFFFF"), width=2),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(
            title="Trading Days (Test Set)",
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Price (USD)",
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        dragmode=False,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#0A192F",
        paper_bgcolor="#0A192F",
        height=450,
    )
    return fig


def plotly_single_model_chart(x_axis, actual, pred, model_name, color):
    """Plotly chart for a single model tab."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=actual, mode="lines", name="Actual",
        line=dict(color=MODEL_COLORS["Actual"], dash="dash", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=pred, mode="lines", name=f"{model_name} Predicted",
        line=dict(color=color, width=2.5),
    ))
    fig.update_layout(
        title=dict(text=f"{model_name} Forecast vs Actual",
                   font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(
            title="Trading Days",
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Price (USD)",
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        dragmode=False,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#0A192F",
        paper_bgcolor="#0A192F",
        height=420,
    )
    return fig


def plotly_residual_chart(x_axis, residuals, model_name, pos_color, neg_color):
    """Plotly bar chart for residual analysis."""
    colors = [pos_color if r >= 0 else neg_color for r in residuals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_axis, y=residuals, marker_color=colors, opacity=0.85,
        name="Residual",
    ))
    fig.add_hline(y=0, line_color="#F8FAFC", line_width=1)
    fig.update_layout(
        title=dict(text=f"Residual Analysis ({model_name})",
                   font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(
            title="Trading Days",
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Actual − Predicted",
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        dragmode=False,
        template="plotly_dark",
        plot_bgcolor="#0A192F",
        paper_bgcolor="#0A192F",
        height=350,
    )
    return fig


def plotly_comparison_bar_chart(summary_df):
    """Interactive Plotly grouped bar chart for all-companies comparison."""
    mdf = summary_df.melt(
        id_vars=["Ticker", "Model"],
        value_vars=["MAE", "RMSE", "MAPE"],
        var_name="Metric",
        value_name="Value",
    )
    fig = px.bar(
        mdf, x="Ticker", y="Value", color="Model",
        barmode="group", facet_col="Metric",
        color_discrete_map=MODEL_COLORS,
        height=500, template="plotly_dark",
    )
    fig.update_layout(
        legend_title_text="Model",
        plot_bgcolor="#0A192F",
        paper_bgcolor="#0A192F",
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode=False,
    )
    return fig


# ─── Data helpers (cached) ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_prepared_data(ticker: str):
    df = load_ticker(ticker, cfg.START_DATE, cfg.END_DATE)
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_data(df)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, cfg.SEQUENCE_LEN)
    return df, X_train, X_test, y_train, y_test, x_scaler, y_scaler, X_test_seq, y_test_seq


@st.cache_resource(show_spinner=False)
def get_saved_models(ticker: str):
    from models import load_models
    return load_models(ticker)


def inverse_scale(y_scaler, values):
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).flatten()


def model_files_exist(ticker: str) -> bool:
    folder = os.path.join("saved_models", ticker)
    return all(
        os.path.exists(os.path.join(folder, name))
        for name in ["rfr.pkl", "ann.keras", "lstm.keras"]
    )


@st.cache_data(show_spinner=False)
def load_metrics_summary():
    path = os.path.join(cfg.OUTPUT_DIR, "metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ─── Universal AI Data Ingestion & Auto-Feature Engine ──────────────────────

def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Cleans and converts any pandas Series into a numeric float series, automatically
    handling currency symbols ($ € £ ¥ ₹), commas, percentage signs, accounting
    parentheses for negatives, and unit suffixes (K, M, B, T).
    """
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    def _parse_val(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s or s.lower() in ["nan", "null", "none", "-", "--", "n/a", "na", "inf", "-inf"]:
            return np.nan
        # Strip currency symbols and extraneous whitespace
        s = s.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace("₹", "")
        s = s.replace(",", "").replace("%", "").strip()
        # Handle accounting parentheses for negative values e.g. (12.34) -> -12.34
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1].strip()
        # Handle suffix multipliers (K, M, B, T)
        multiplier = 1.0
        if s.endswith(("K", "k")):
            multiplier = 1e3
            s = s[:-1].strip()
        elif s.endswith(("M", "m")):
            multiplier = 1e6
            s = s[:-1].strip()
        elif s.endswith(("B", "b")):
            multiplier = 1e9
            s = s[:-1].strip()
        elif s.endswith(("T", "t")):
            multiplier = 1e12
            s = s[:-1].strip()
        try:
            val_float = float(s) * multiplier
            return np.nan if np.isinf(val_float) else val_float
        except (ValueError, TypeError):
            return np.nan

    cleaned = series.apply(_parse_val)
    return cleaned.replace([np.inf, -np.inf], np.nan)


def _load_any_file(uploaded_file) -> pd.DataFrame:
    """
    Universally loads any file format (CSV, TSV, TXT, Excel, JSON, Parquet)
    into a pandas DataFrame with automatic delimiter & encoding fallback.
    """
    fname = getattr(uploaded_file, "name", "").lower()
    df = None

    # 1. Parquet
    if fname.endswith(".parquet"):
        try:
            uploaded_file.seek(0)
            return pd.read_parquet(uploaded_file)
        except Exception as e:
            pass

    # 2. Excel (.xlsx, .xls)
    if fname.endswith((".xlsx", ".xls")):
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
        except Exception:
            pass

    # 3. JSON / JSONL
    if fname.endswith((".json", ".jsonl", ".ndjson")):
        try:
            uploaded_file.seek(0)
            return pd.read_json(uploaded_file)
        except Exception:
            try:
                uploaded_file.seek(0)
                return pd.read_json(uploaded_file, lines=True)
            except Exception:
                pass

    # 4. Delimited Text (CSV, TSV, TXT, custom separators)
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding=enc)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    # Fallback to standard delimiters
    for sep in [",", "\t", ";", "|", r"\s+"]:
        for enc in ["utf-8", "latin1"]:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, encoding=enc)
                if df is not None and not df.empty and len(df.columns) > 0:
                    return df
            except Exception:
                continue

    if df is None or df.empty:
        raise ValueError(
            "Could not parse uploaded file. Please ensure it is a valid CSV, Excel, "
            "JSON, Parquet, or TSV file."
        )
    return df


def prepare_uploaded_data(uploaded_file, selected_target=None):
    """
    AI-driven 100% universal data ingestion and auto-feature engineering engine.
    Accepts ANY time-series data from ANY domain (Finance, Weather, Energy, IoT, Sales, Web).
    Automatically identifies date indices, handles missing features, synthesizes technical
    indicators, and allows forecasting of ANY numeric target column.
    
    Returns (clean_df, detection_report, candidate_numeric_cols, chosen_target).
    """
    from data_loader import _sma, _rsi

    raw_df = _load_any_file(uploaded_file)
    if raw_df.empty:
        raise ValueError("Uploaded dataset is empty.")

    df = raw_df.copy()

    # Flatten multi-level column names if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Large dataset optimization: retain latest 50,000 records for responsive modeling
    max_rows = 50_000
    is_truncated = False
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()
        is_truncated = True

    detection_report = {}
    if is_truncated:
        detection_report["Data Scope"] = f"Using latest {max_rows:,} rows (from {len(raw_df):,} total)"

    # 1. Universal Date / Timestamp Detection
    date_col = None
    date_keywords = [
        "date", "datetime", "timestamp", "time", "day", "period", "dates",
        "trade_date", "tradedate", "dt", "d", "trans_date", "quote_date",
        "year", "month", "hour", "step", "epoch"
    ]

    if isinstance(df.index, pd.DatetimeIndex):
        detection_report["Temporal Index"] = "Existing Datetime Index"
    else:
        # Check columns matching date keywords
        for col in df.columns:
            clean_col = str(col).strip().lower().replace("_", "").replace(" ", "").replace("-", "")
            if any(dk in clean_col for dk in date_keywords):
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notna().sum() > len(df) * 0.4:
                        df[col] = converted
                        date_col = col
                        break
                except Exception:
                    continue

        # Check numeric columns for Unix epoch timestamps
        if date_col is None:
            for col in df.select_dtypes(include=[np.number]).columns:
                try:
                    sample = df[col].dropna()
                    if not sample.empty:
                        min_val, max_val = sample.min(), sample.max()
                        # Epoch seconds
                        if 946684800 <= min_val and max_val <= 2524608000:
                            df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
                            date_col = col
                            break
                        # Epoch milliseconds
                        elif 946684800000 <= min_val and max_val <= 2524608000000:
                            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
                            date_col = col
                            break
                except Exception:
                    continue

        # Check object/string columns for convertible datetime
        if date_col is None:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notna().sum() > len(df) * 0.5:
                        df[col] = converted
                        date_col = col
                        break
                except Exception:
                    continue

        if date_col is not None:
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            detection_report["Temporal Index"] = f"'{date_col}' (Auto-detected)"
        else:
            # Generate chronological business-day sequence
            df.index = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="B")
            detection_report["Temporal Index"] = "Auto-generated Chronological Sequence"

    # 2. Universal Numeric Cleaning across all columns
    cleaned_series_dict = {}
    valid_numeric_cols = []
    for col in df.columns:
        s = _clean_numeric_series(df[col])
        if s.notna().sum() >= min(3, len(df)):
            cleaned_series_dict[col] = s
            valid_numeric_cols.append(col)

    if not valid_numeric_cols:
        raise ValueError("No numeric data columns found in the uploaded file.")

    # 3. Intelligent Target Variable Selection
    domain_priority_aliases = [
        # Finance & Markets
        "close", "adjclose", "adj_close", "adjustedclose", "price", "last", "ltp", "rate", "nav", "settle", "value",
        # Energy & Utilities
        "demand", "load", "consumption", "power", "usage", "mw", "kwh", "energy",
        # Weather, Science & IoT
        "temperature", "temp", "humidity", "pressure", "rainfall", "wind", "speed", "vibration", "voltage", "reading",
        # Business & Web Analytics
        "sales", "revenue", "traffic", "visits", "users", "orders", "count", "quantity", "units",
        # Generic Targets
        "target", "y", "output", "signal", "val", "result"
    ]

    chosen_target = None
    if selected_target and selected_target in valid_numeric_cols:
        chosen_target = selected_target
        detection_report["Target Variable"] = f"'{chosen_target}' (User-Selected)"
    else:
        # Search priority aliases
        for alias in domain_priority_aliases:
            for col in valid_numeric_cols:
                clean_col = str(col).strip().lower().replace("_", "").replace(" ", "").replace("-", "")
                if alias == clean_col or alias in clean_col:
                    chosen_target = col
                    break
            if chosen_target is not None:
                break

        # Fallback: pick the numeric column with the highest variance
        if chosen_target is None:
            chosen_target = max(
                valid_numeric_cols,
                key=lambda c: cleaned_series_dict[c].std(skipna=True) if cleaned_series_dict[c].std(skipna=True) > 0 else 0
            )
            detection_report["Target Variable"] = f"'{chosen_target}' (Auto-detected Primary Metric)"
        else:
            detection_report["Target Variable"] = f"'{chosen_target}' (Auto-detected)"

    # 4. Universal Feature Mapping & Synthesis for Models (Open, High, Low, Close, Volume)
    clean_df = pd.DataFrame(index=df.index)
    clean_df["Close"] = cleaned_series_dict[chosen_target]

    # Remaining columns excluding chosen target
    other_cols = [c for c in valid_numeric_cols if c != chosen_target]

    # Open: alias match or lag-1 synthesis
    open_col = next((c for c in other_cols if any(k in c.lower() for k in ["open", "start", "first"])), None)
    if open_col:
        clean_df["Open"] = cleaned_series_dict[open_col]
        detection_report["Open"] = f"'{open_col}'"
    else:
        clean_df["Open"] = clean_df["Close"].shift(1).bfill().ffill()
        detection_report["Open"] = "Auto-synthesized (Lag-1 Trend)"

    # High: alias match or dynamic peak envelope
    high_col = next((c for c in other_cols if any(k in c.lower() for k in ["high", "max", "peak", "top"])), None)
    if high_col:
        clean_df["High"] = cleaned_series_dict[high_col]
        detection_report["High"] = f"'{high_col}'"
    else:
        clean_df["High"] = np.maximum(clean_df["Open"], clean_df["Close"]) * 1.002
        detection_report["High"] = "Auto-synthesized (Local Max +0.2%)"

    # Low: alias match or dynamic trough envelope
    low_col = next((c for c in other_cols if any(k in c.lower() for k in ["low", "min", "bottom"])), None)
    if low_col:
        clean_df["Low"] = cleaned_series_dict[low_col]
        detection_report["Low"] = f"'{low_col}'"
    else:
        clean_df["Low"] = np.minimum(clean_df["Open"], clean_df["Close"]) * 0.998
        detection_report["Low"] = "Auto-synthesized (Local Min -0.2%)"

    # Volume: alias match or dynamic activity proxy
    vol_col = next((c for c in other_cols if any(k in c.lower() for k in ["vol", "qty", "count", "trade", "shares"])), None)
    if vol_col:
        clean_df["Volume"] = cleaned_series_dict[vol_col]
        detection_report["Volume"] = f"'{vol_col}'"
    else:
        pct_diff = clean_df["Close"].pct_change().abs().fillna(0)
        clean_df["Volume"] = (pct_diff + 1.0) * 1_000_000.0
        detection_report["Volume"] = "Auto-synthesized (Activity Proxy)"

    # 5. Missing Data Imputation & Sanitization
    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_df.interpolate(method="linear", limit_direction="both", inplace=True)
    clean_df.ffill(inplace=True)
    clean_df.bfill(inplace=True)
    clean_df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)

    if len(clean_df) < 5:
        raise ValueError(
            f"Dataset requires at least 5 valid sequential data points, "
            f"got {len(clean_df)}."
        )

    # 6. Feature Engineering: SMA, RSI, Target
    clean_df["SMA"] = _sma(clean_df["Close"], cfg.SMA_WINDOW)
    clean_df["RSI"] = _rsi(clean_df["Close"], cfg.RSI_WINDOW)
    clean_df["Target"] = clean_df["Close"].shift(-1).bfill()
    clean_df.dropna(inplace=True)

    detection_report["SMA & RSI"] = "Auto-computed Moving Average & Momentum"
    detection_report["Target Feature"] = "Auto-generated Next-Step Ahead Target"

    return clean_df, detection_report, valid_numeric_cols, chosen_target


# ─── All Companies Comparison (Plotly version) ──────────────────────────────

def show_all_companies_comparison():
    st.markdown(
        '<h2 style="color:#00C896;text-align:center;font-weight:700;margin:8px 0">'
        '📊 All Companies Comparison</h2>',
        unsafe_allow_html=True,
    )
    summary_df = load_metrics_summary()
    if summary_df is None or summary_df.empty:
        st.info("Run `python main.py` to create `outputs/metrics.csv` and calculate the overall winner.")
        return

    # Interactive Plotly chart
    st.plotly_chart(
        plotly_comparison_bar_chart(summary_df),
        use_container_width=True,
        config=COMPARISON_PLOTLY_CONFIG,
    )
    st.caption(
        "This graph compares ANN, Random Forest, and LSTM for every company. "
        "Lower MAE, RMSE, and MAPE values mean better prediction accuracy."
    )

    # Best model summary
    avg_metrics = (
        summary_df.groupby("Model")[["MAE", "RMSE", "MAPE"]]
        .mean()
        .sort_values("RMSE")
    )
    best_model = avg_metrics.index[0]
    best_mae = avg_metrics.loc[best_model, "MAE"]
    best_rmse = avg_metrics.loc[best_model, "RMSE"]
    best_mape = avg_metrics.loc[best_model, "MAPE"]

    st.success(
        f"🏆 Best overall model: **{best_model}**. Average MAE: {best_mae:.2f}, "
        f"average RMSE: {best_rmse:.2f}, average MAPE: {best_mape:.2f}%."
    )

    # Styled dataframe
    styled = avg_metrics.reset_index().style.format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv_data = summary_df.to_csv(index=False).encode()
    st.download_button(
        label="💾 Download full metrics CSV",
        data=csv_data,
        file_name="stocksight_metrics.csv",
        mime="text/csv",
    )


# ─── Custom CSV / Data Studio Page ──────────────────────────────────────────

def show_upload_studio(show_ann, show_rfr, show_lstm):
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); 
                    padding: 24px; border-radius: 12px; border: 1px solid #334155; 
                    margin-bottom: 24px; text-align: center;">
            <h2 style="color:#00C896; margin:0 0 8px 0; font-weight:700;">📂 Custom Dataset & CSV Studio</h2>
            <p style="color:#94A3B8; margin:0; font-size:15px;">
                Upload any time-series dataset from any domain (Stock, Crypto, Weather, Sales, Energy, IoT). 
                AI auto-detects columns, repairs missing features, and runs multi-model forecasts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "📁 Drag and drop or browse any CSV, Excel, JSON, Parquet, or TSV file (Up to 1 GB)",
        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "parquet", "ndjson"],
        help="Upload data from any source. Minimum 5 sequential data points required.",
        key="main_studio_uploader",
    )

    if uploaded_file is None:
        st.info("👈 **Get Started**: Upload any CSV or time-series data file above to generate forecasts.")
        
        # Supported features grid
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div style="background:#1E293B; padding:16px; border-radius:8px; border:1px solid #334155;">
                    <h4 style="color:#00C896; margin-top:0;">⚡ Universal Formats</h4>
                    <p style="color:#CBD5E1; font-size:13px; margin-bottom:0;">
                        Accepts <code>.csv</code>, <code>.xlsx</code>, <code>.json</code>, <code>.parquet</code>, and <code>.tsv</code> with automatic delimiter & encoding recognition.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div style="background:#1E293B; padding:16px; border-radius:8px; border:1px solid #334155;">
                    <h4 style="color:#00C896; margin-top:0;">🤖 100% Auto-Mapping</h4>
                    <p style="color:#CBD5E1; font-size:13px; margin-bottom:0;">
                        Automatically matches 25+ aliases for Date, Price, Open, High, Low, and Volume. Synthesizes any missing columns.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div style="background:#1E293B; padding:16px; border-radius:8px; border:1px solid #334155;">
                    <h4 style="color:#00C896; margin-top:0;">🎯 Any Target Column</h4>
                    <p style="color:#CBD5E1; font-size:13px; margin-bottom:0;">
                        Forecast stock prices, energy consumption, temperatures, sales counts, or any custom numeric variable.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    try:
        # Pre-process data
        target_override = st.session_state.get("custom_target_override", None)
        custom_df, detection_report, candidate_cols, chosen_target = prepare_uploaded_data(
            uploaded_file, selected_target=target_override
        )

        # Allow user to switch target variable if multiple numeric columns exist
        if len(candidate_cols) > 1:
            col_sel1, col_sel2 = st.columns([2, 3])
            with col_sel1:
                target_choice = st.selectbox(
                    "🎯 Forecast Target Variable (Select any column):",
                    options=candidate_cols,
                    index=candidate_cols.index(chosen_target) if chosen_target in candidate_cols else 0,
                    help="Choose which numerical variable you want the models to forecast.",
                )
                if target_choice != chosen_target:
                    st.session_state["custom_target_override"] = target_choice
                    st.rerun()

        X_train_u, X_test_u, y_train_u, y_test_u, x_scaler_u, y_scaler_u = prepare_data(custom_df)
        X_full_scaled = np.vstack([X_train_u, X_test_u])

        # Load models
        loaded_ticker = None
        for t in cfg.TICKERS:
            if model_files_exist(t):
                loaded_ticker = t
                break

        if loaded_ticker is None:
            st.error("No saved models found. Run `python main.py` first.")
            return

        rfr, ann, lstm = get_saved_models(loaded_ticker)

        seq_len = cfg.SEQUENCE_LEN
        n_train = len(X_train_u)
        n_test = len(X_test_u)
        lstm_test_seqs = []

        if n_test > seq_len:
            offset = seq_len
            actual = inverse_scale(y_scaler_u, y_test_u[offset:])
            ann_input = X_test_u[offset:]
            rfr_input = X_test_u[offset:]
            for i in range(offset, n_test):
                lstm_test_seqs.append(X_test_u[i - seq_len : i])
        else:
            offset = 0
            actual = inverse_scale(y_scaler_u, y_test_u)
            ann_input = X_test_u
            rfr_input = X_test_u
            for k in range(n_test):
                idx = n_train + k
                lookback = X_full_scaled[max(0, idx - seq_len) : idx]
                if len(lookback) < seq_len:
                    padded = np.pad(lookback, ((seq_len - len(lookback), 0), (0, 0)), mode="edge")
                else:
                    padded = lookback
                lstm_test_seqs.append(padded)

        X_test_seq_u = np.array(lstm_test_seqs)
        x_axis = list(range(len(actual)))

        with st.spinner(f"Running forecasting models for {chosen_target}..."):
            ann_pred = inverse_scale(y_scaler_u, ann.predict(ann_input, verbose=0).flatten())
            rfr_pred = inverse_scale(y_scaler_u, rfr.predict(rfr_input))
            lstm_pred = inverse_scale(y_scaler_u, lstm.predict(X_test_seq_u, verbose=0).flatten())

        predictions = {"ANN": ann_pred, "RFR": rfr_pred, "LSTM": lstm_pred}
        show_flags = {"ANN": show_ann, "RFR": show_rfr, "LSTM": show_lstm}

        # AI Column & Domain Ingestion Card
        with st.expander("🤖 AI Universal Data Ingestion Report (Auto-Processed)", expanded=True):
            cols_per_row = 3
            items = list(detection_report.items())
            for i in range(0, len(items), cols_per_row):
                row_items = items[i : i + cols_per_row]
                row_cols = st.columns(len(row_items))
                for col_widget, (k, v) in zip(row_cols, row_items):
                    col_widget.metric(label=f"📌 {k}", value=v)

        # Tabbed Results Navigation
        tab_chart, tab_metrics, tab_data = st.tabs([
            "📈 Forecast vs Actual",
            "📋 Performance Metrics",
            "🔍 Dataset & Features Explorer",
        ])

        with tab_chart:
            fig = plotly_prediction_chart(
                x_axis, actual, predictions,
                f"Multi-Model Forecast vs Actual ({chosen_target})",
                show_flags=show_flags,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_metrics:
            st.subheader(f"📋 Model Evaluation Metrics ({chosen_target})")
            rows = []
            for name, pred in predictions.items():
                if not show_flags.get(name, True):
                    continue
                m = metrics(actual, pred)
                rows.append({
                    "Model": name,
                    "MAE": f"{m['MAE']:.4f}",
                    "RMSE": f"{m['RMSE']:.4f}",
                    "MAPE": f"{m['MAPE']:.2f}%",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            pred_df = pd.DataFrame({"Actual": actual})
            for name, pred in predictions.items():
                pred_df[name] = pred
            csv_data = pred_df.to_csv(index=False).encode()
            st.download_button(
                label=f"💾 Download {chosen_target} Predictions CSV",
                data=csv_data,
                file_name=f"{chosen_target}_predictions.csv",
                mime="text/csv",
            )

        with tab_data:
            st.subheader("🔍 Auto-Engineered 7-Feature Model Matrix")
            st.caption(f"Total Rows: **{len(custom_df):,}** | Primary Target: **{chosen_target}**")
            st.dataframe(custom_df.head(20), use_container_width=True)

            st.subheader("📊 Statistical Summary")
            st.dataframe(custom_df.describe().T, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process uploaded file: {e}")


# ─── Hero Banner ────────────────────────────────────────────────────────────

with st.container():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📈 StockSight</h1>
            <p>Multi‑Model Stock & Time-Series Forecasting Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Enhanced Sidebar Navigation ────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<h3 style="color:#00C896;text-align:center;">🧭 Navigation</h3>',
        unsafe_allow_html=True,
    )

    app_section = st.radio(
        "Choose Section",
        [
            "📈 Stock Market Analysis",
            "📂 Custom CSV / Data Studio",
            "📊 All Companies Comparison",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    show_ann = True
    show_rfr = True
    show_lstm = True

    st.divider()

    if app_section == "📈 Stock Market Analysis":
        ticker = st.selectbox(
            "📌 Select Stock",
            cfg.TICKERS,
            index=0,
            placeholder="Choose a stock...",
        )
        load_dashboard = st.button(
            "🚀 Load Stock Forecasts",
            type="primary",
            use_container_width=True,
        )
    else:
        ticker = None
        load_dashboard = True

    # ---- Info captions ----
    st.caption(f"📅 Stock Period: {cfg.START_DATE} → {cfg.END_DATE}")
    st.caption("📊 Split: 80 % Train / 20 % Test")
    st.caption(f"🔄 LSTM Lookback: {cfg.SEQUENCE_LEN} steps")




# ─── Main content routing ───────────────────────────────────────────────────

if app_section == "📊 All Companies Comparison":
    show_all_companies_comparison()
    st.stop()

elif app_section == "📂 Custom CSV / Data Studio":
    show_upload_studio(show_ann, show_rfr, show_lstm)
    st.stop()

# ---- Pre-Trained Stock Analysis Section ----
col1 = st.columns(1)[0]
col1.metric("Selected Stock", ticker or "Not selected")

if ticker is None:
    st.info("👈 Choose a Stock in the sidebar to continue.")
    st.stop()

if not model_files_exist(ticker):
    st.error(f"No saved models found for {ticker}. Run `python main.py` first.")
    st.stop()


with st.spinner("Loading Data and Saved Models..."):
    try:
        (
            df,
            X_train,
            X_test,
            y_train,
            y_test,
            x_scaler,
            y_scaler,
            X_test_seq,
            y_test_seq,
        ) = get_prepared_data(ticker)
        rfr, ann, lstm = get_saved_models(ticker)
    except Exception as exc:
        st.error(f"Could not load dashboard: {exc}")
        st.stop()


offset = cfg.SEQUENCE_LEN
actual = inverse_scale(y_scaler, y_test[offset:])
x_axis = list(range(len(actual)))

pred_csv_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_predictions.csv")

if os.path.exists(pred_csv_path):
    with st.spinner("Loading Saved Predictions..."):
        pred_df = pd.read_csv(pred_csv_path)
        ann_pred = pred_df["ANN"].values
        rfr_pred = pred_df["RFR"].values
        lstm_pred = pred_df["LSTM"].values
else:
    with st.spinner("Running Predictions (Fallback)..."):
        ann_pred = inverse_scale(y_scaler, ann.predict(X_test[offset:], verbose=0).flatten())
        rfr_pred = inverse_scale(y_scaler, rfr.predict(X_test[offset:]))
        lstm_pred = inverse_scale(y_scaler, lstm.predict(X_test_seq, verbose=0).flatten())

predictions = {
    "ANN": ann_pred,
    "RFR": rfr_pred,
    "LSTM": lstm_pred,
}
show_flags = {"ANN": show_ann, "RFR": show_rfr, "LSTM": show_lstm}


# Create tabs for structured navigation
tab_comp, tab_ann, tab_rfr, tab_lstm = st.tabs([
    "📈 Comparison (All Models)",
    "🧠 ANN Model",
    "🌲 RFR Model",
    "🔄 LSTM Model"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: COMPARISON (ALL MODELS VS ACTUAL)  — now with Plotly
# ─────────────────────────────────────────────────────────────────────────────
with tab_comp:
    st.subheader(f"All Models vs Actual Price for {ticker}")

    fig = plotly_prediction_chart(
        x_axis, actual, predictions,
        f"All Models vs Actual — {ticker}",
        show_flags=show_flags,
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    **Graph Explanation:**
    This comparison chart plots the actual stock closing prices alongside predictions from all three models (ANN, Random Forest, and LSTM) over the testing period.
    - The **Actual** price is represented by the **dashed grey line**.
    - The **ANN (Artificial Neural Network)** prediction is the **green line**.
    - The **RFR (Random Forest Regressor)** prediction is the **blue line**.
    - The **LSTM (Long Short-Term Memory)** prediction is the **purple line**.

    *Look for which model line tracks the actual price line most closely, especially during sharp trends or sudden reversals.*

    💡 **Tip:** Hover over the chart to see exact values. Use mouse-wheel or zoom/pan tools in the top-right toolbar.
    """)

    st.subheader("📋 Performance Metrics Summary Table")
    rows = []
    for name, pred in predictions.items():
        if not show_flags.get(name, True):
            continue
        model_metrics = metrics(actual, pred)
        rows.append(
            {
                "Model": name,
                "MAE (Mean Absolute Error)": f"${model_metrics['MAE']:.2f}",
                "RMSE (Root Mean Squared Error)": f"${model_metrics['RMSE']:.2f}",
                "MAPE (Mean Absolute Percentage Error)": f"{model_metrics['MAPE']:.2f}%",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("""
    **Metrics Explanation:**
    - **MAE (Mean Absolute Error):** On average, how many dollars the predictions deviate from actual values. Lower is better.
    - **RMSE (Root Mean Squared Error):** Similiar to MAE, but penalizes larger errors more heavily. Lower is better.
    - **MAPE (Mean Absolute Percentage Error):** The average percentage deviation of the predictions from actual prices. Lower is better.
    """)

    # Download metrics
    if rows:
        csv_data = pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button(
            label="💾 Download metrics CSV",
            data=csv_data,
            file_name=f"{ticker}_metrics.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANN MODEL  — now with Plotly
# ─────────────────────────────────────────────────────────────────────────────
with tab_ann:
    st.subheader(f"ANN Model vs Actual - {ticker}")

    m1, m2, m3 = st.columns(3)
    ann_metrics = metrics(actual, predictions["ANN"])
    m1.metric("ANN MAE", f"${ann_metrics['MAE']:.2f}")
    m2.metric("ANN RMSE", f"${ann_metrics['RMSE']:.2f}")
    m3.metric("ANN MAPE", f"{ann_metrics['MAPE']:.2f}%")

    fig = plotly_single_model_chart(x_axis, actual, predictions["ANN"], "ANN", MODEL_COLORS["ANN"])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    **Graph Explanation:**
    This graph displays the actual stock price (dashed line) versus the Artificial Neural Network (ANN) predictions (green line). The ANN uses dense feed-forward connections to map features to prices, indicating how well simple neural mapping captures daily stock trends.
    """)

    st.subheader("Residual Analysis (ANN)")
    residuals = actual - predictions["ANN"]
    fig = plotly_residual_chart(x_axis, residuals, "ANN", "#059669", "#DC2626")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("""
    **Residuals Explanation:**
    Residuals represent the prediction error (Actual minus Predicted). Green bars indicate the model under-predicted (actual was higher), while red bars indicate over-prediction (actual was lower). The ideal model has small bars hovering close to zero.
    """)

    st.subheader("Training vs Validation Loss Curve")
    loss_curve_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_ANN_loss_curve.png")
    if os.path.exists(loss_curve_path):
        st.image(loss_curve_path, use_container_width=True)
        st.markdown("""
        **Loss Curve Explanation:**
        The training vs. validation loss curve shows how the neural network learned over training epochs. A healthy training run features both training loss (solid line) and validation loss (dashed line) declining steadily and converging. If validation loss increases while training loss decreases, the model is overfitting.
        """)
    else:
        st.warning(f"No Saved Loss Curve Found at `{loss_curve_path}`.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: RFR MODEL  — now with Plotly
# ─────────────────────────────────────────────────────────────────────────────
with tab_rfr:
    st.subheader(f"RFR Model vs Actual - {ticker}")

    m1, m2, m3 = st.columns(3)
    rfr_metrics = metrics(actual, predictions["RFR"])
    m1.metric("RFR MAE", f"${rfr_metrics['MAE']:.2f}")
    m2.metric("RFR RMSE", f"${rfr_metrics['RMSE']:.2f}")
    m3.metric("RFR MAPE", f"{rfr_metrics['MAPE']:.2f}%")

    fig = plotly_single_model_chart(x_axis, actual, predictions["RFR"], "RFR", MODEL_COLORS["RFR"])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    **Graph Explanation:**
    This graph displays the actual stock price (dashed line) versus the Random Forest Regressor (RFR) predictions (blue line). RFR uses an ensemble of decision trees, which are less prone to overfitting but may predict in steps rather than smooth curves.
    """)

    st.subheader("Residual Analysis (RFR)")
    residuals = actual - predictions["RFR"]
    fig = plotly_residual_chart(x_axis, residuals, "RFR", "#059669", "#DC2626")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("""
    **Residuals Explanation:**
    Residuals represent the prediction error (Actual minus Predicted). Green bars indicate the model under-predicted, while red bars indicate over-prediction.
    """)

    st.subheader("Feature Importance Plot")
    feature_importance_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_feature_importance.png")
    if os.path.exists(feature_importance_path):
        st.image(feature_importance_path, use_container_width=True)
        st.markdown("""
        **Feature Importance Explanation:**
        This chart ranks the engineered features by their predictive importance in the Random Forest model. Higher scores indicate the model relied more heavily on that feature (e.g., SMA, Close price, Volume) to make its decisions.
        """)
    else:
        st.warning(f"No Saved Feature-Importance Plot Found at `{feature_importance_path}`. Run `python main.py` to generate it.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: LSTM MODEL  — now with Plotly
# ─────────────────────────────────────────────────────────────────────────────
with tab_lstm:
    st.subheader(f"LSTM Model vs Actual - {ticker}")

    m1, m2, m3 = st.columns(3)
    lstm_metrics = metrics(actual, predictions["LSTM"])
    m1.metric("LSTM MAE", f"${lstm_metrics['MAE']:.2f}")
    m2.metric("LSTM RMSE", f"${lstm_metrics['RMSE']:.2f}")
    m3.metric("LSTM MAPE", f"{lstm_metrics['MAPE']:.2f}%")

    fig = plotly_single_model_chart(x_axis, actual, predictions["LSTM"], "LSTM", MODEL_COLORS["LSTM"])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    **Graph Explanation:**
    This graph displays the actual stock price (dashed line) versus the Long Short-Term Memory (LSTM) network predictions (purple line). LSTM is a recurrent neural network designed for sequence processing, allowing it to capture historical price trends and momentum.
    """)

    st.subheader("Residual Analysis (LSTM)")
    residuals = actual - predictions["LSTM"]
    fig = plotly_residual_chart(x_axis, residuals, "LSTM", "#059669", "#DC2626")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("""
    **Residuals Explanation:**
    Residuals represent the prediction error (Actual minus Predicted). Green bars indicate the model under-predicted, while red bars indicate over-prediction.
    """)

    st.subheader("Training vs Validation Loss Curve")
    loss_curve_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_LSTM_loss_curve.png")
    if os.path.exists(loss_curve_path):
        st.image(loss_curve_path, use_container_width=True)
        st.markdown("""
        **Loss Curve Explanation:**
        The training vs. validation loss curve shows how the neural network learned over training epochs. A healthy training run features both training loss (solid line) and validation loss (dashed line) declining steadily and converging. If validation loss increases while training loss decreases, the model is overfitting.
        """)
    else:
        st.warning(f"No Saved Loss Curve Found at `{loss_curve_path}`.")


# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        &copy; 2026 <strong>StockSight</strong> &mdash; Multi-Model Financial &amp; Time-Series Forecasting Platform
    </div>
    """,
    unsafe_allow_html=True,
)

