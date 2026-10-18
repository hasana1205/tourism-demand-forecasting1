import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Tourism Demand Forecasting", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#123b55,#168b8b); color: white; margin-bottom: 1rem;}
    .hero h1 {color:white; margin-bottom:.25rem;}
    .hero p {color:#e7f7f7; font-size:1.05rem;}
    div[data-testid="stMetric"] {background:white; padding:1rem; border-radius:14px; border:1px solid #e8edf3;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌍 Tourism Demand Forecasting</h1>
  <p>Explore tourist arrivals, check destination weather, and estimate future demand using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Data ----------------
@st.cache_data
def demo_data():
    rng = np.random.default_rng(42)
    years = np.arange(2015, 2026)
    rows = []
    base = 52000
    for i, year in enumerate(years):
        for month in range(1, 13):
            season = 1 + 0.22 * np.sin((month - 1) / 12 * 2 * np.pi)
            trend = 1 + 0.045 * i
            disruption = 0.62 if year == 2020 else (0.78 if year == 2021 else 1)
            arrivals = max(1000, int(base * season * trend * disruption + rng.normal(0, 3500)))
            rows.append({"Year": int(year), "Month": int(month), "Tourist_Arrivals": arrivals})
    return pd.DataFrame(rows)

st.sidebar.header("⚙️ Dashboard settings")
destination = st.sidebar.selectbox(
    "Destination for weather",
    ["Chennai", "Ooty", "Coimbatore", "Madurai", "Kodaikanal", "Bengaluru", "Delhi", "Mumbai"]
)
model_choice = st.sidebar.selectbox("Prediction model", ["Random Forest", "Linear Regression"])
uploaded = st.sidebar.file_uploader(
    "Upload tourist-arrivals CSV (optional)",
    type=["csv"],
    help="Recommended columns: Year, Month, Tourist_Arrivals"
)

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        required = {"Year", "Month", "Tourist_Arrivals"}
        if not required.issubset(df.columns):
            st.error("CSV must contain these columns: Year, Month, Tourist_Arrivals. Using sample data for now.")
            df = demo_data()
        else:
            df = df[["Year", "Month", "Tourist_Arrivals"]].copy()
            for col in ["Year", "Month", "Tourist_Arrivals"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            df = df[(df["Month"].between(1, 12)) & (df["Tourist_Arrivals"] >= 0)]
            if len(df) < 12:
                st.warning("Please upload at least 12 valid monthly rows. Using sample data for now.")
                df = demo_data()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = demo_data()
else:
    df = demo_data()

df = df.sort_values(["Year", "Month"]).reset_index(drop=True)
df["Date"] = pd.to_datetime(dict(year=df["Year"].astype(int), month=df["Month"].astype(int), day=1))
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
df["Time_Index"] = (df["Year"] - df["Year"].min()) * 12 + df["Month"]

# ---------------- Main dashboard ----------------
total_arrivals = int(df["Tourist_Arrivals"].sum())
average_monthly = int(df["Tourist_Arrivals"].mean())
latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) > 1 else latest
change = ((latest["Tourist_Arrivals"] - previous["Tourist_Arrivals"]) / max(previous["Tourist_Arrivals"], 1)) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total arrivals in dataset", f"{total_arrivals:,}")
c2.metric("Average monthly arrivals", f"{average_monthly:,}")
c3.metric("Latest recorded month", latest["Date"].strftime("%b %Y"))
c4.metric("Change vs previous row", f"{change:+.1f}%")

tab_overview, tab_forecast, tab_weather, tab_chatbot, tab_data = st.tabs(
    ["📊 Overview", "🔮 Forecast", "🌦️ Weather", "💬 Travel chatbot", "🗂️ Dataset"]
)

with tab_overview:
    st.subheader("Tourist arrivals over time")
    fig = px.line(df, x="Date", y="Tourist_Arrivals", markers=True,
                  title="Monthly tourist arrivals", labels={"Tourist_Arrivals": "Tourist arrivals", "Date": "Month"})
    fig.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        month_avg = df.groupby("Month", as_index=False)["Tourist_Arrivals"].mean()
        month_avg["Month_Name"] = month_avg["Month"].map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        )
        fig_month = px.bar(month_avg, x="Month_Name", y="Tourist_Arrivals",
                           title="Average arrivals by month", labels={"Tourist_Arrivals":"Average arrivals","Month_Name":"Month"})
        fig_month.update_layout(template="plotly_white")
        st.plotly_chart(fig_month, use_container_width=True)
    with right:
        year_avg = df.groupby("Year", as_index=False)["Tourist_Arrivals"].sum()
        fig_year = px.bar(year_avg, x="Year", y="Tourist_Arrivals",
                          title="Total arrivals by year", labels={"Tourist_Arrivals":"Total arrivals"})
        fig_year.update_layout(template="plotly_white")
        st.plotly_chart(fig_year, use_container_width=True)

with tab_forecast:
    st.subheader("Machine-learning forecast")
    st.caption("The built-in dataset is simulated demo data. Upload verified historical tourist arrivals for meaningful project results.")
    horizon = st.slider("How many months ahead?", min_value=1, max_value=24, value=6)
    features = ["Year", "Month", "Month_Sin", "Month_Cos", "Time_Index"]
    X = df[features]
    y = df["Tourist_Arrivals"]
    if len(df) >= 12:
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
        else:
            model = LinearRegression()
        # Hold out the latest 20% in time order for a simple evaluation.
        split = max(1, int(len(df) * 0.8))
        if split < len(df) and split >= 6:
            eval_model = (RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
                          if model_choice == "Random Forest" else LinearRegression())
            eval_model.fit(X.iloc[:split], y.iloc[:split])
            eval_pred = eval_model.predict(X.iloc[split:])
            mae = mean_absolute_error(y.iloc[split:], np.maximum(0, eval_pred))
            st.metric("Holdout mean absolute error (MAE)", f"{mae:,.0f} arrivals")
        model.fit(X, y)
        last_date = df["Date"].max()
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        future = pd.DataFrame({"Date": future_dates})
        future["Year"] = future["Date"].dt.year
        future["Month"] = future["Date"].dt.month
        future["Month_Sin"] = np.sin(2 * np.pi * future["Month"] / 12)
        future["Month_Cos"] = np.cos(2 * np.pi * future["Month"] / 12)
        future["Time_Index"] = (future["Year"] - df["Year"].min()) * 12 + future["Month"]
        future["Predicted_Arrivals"] = np.maximum(0, model.predict(future[features])).round().astype(int)
        historical = df[["Date", "Tourist_Arrivals"]].rename(columns={"Tourist_Arrivals":"Arrivals"})
        future_plot = future[["Date", "Predicted_Arrivals"]].rename(columns={"Predicted_Arrivals":"Arrivals"})
        historical["Type"] = "Historical"
        future_plot["Type"] = "Forecast"
        combined = pd.concat([historical, future_plot.assign(Type="Forecast")], ignore_index=True)
        fig = px.line(combined, x="Date", y="Arrivals", color="Type", markers=True,
                      title=f"{model_choice} — historical data and forecast")
        fig.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(future[["Date", "Predicted_Arrivals"]].rename(
            columns={"Date":"Forecast month", "Predicted_Arrivals":"Estimated tourist arrivals"}
        ), use_container_width=True, hide_index=True)
        st.download_button("Download forecast CSV", future.to_csv(index=False).encode("utf-8"),
                           file_name="tourism_forecast.csv", mime="text/csv")
    else:
        st.warning("At least 12 monthly observations are required to train a basic forecast model.")

with tab_weather:
    st.subheader(f"Current weather — {destination}")
    st.write("Weather data uses the Open-Meteo API, which does not require an API key.")
    coords = {
        "Chennai": (13.0827, 80.2707), "Ooty": (11.4102, 76.6950),
        "Coimbatore": (11.0168, 76.9558), "Madurai": (9.9252, 78.1198),
        "Kodaikanal": (10.2381, 77.4892), "Bengaluru": (12.9716, 77.5946),
        "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777)
    }
    lat, lon = coords[destination]
    if st.button("Get current weather", type="primary"):
        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"},
                timeout=12
            )
            response.raise_for_status()
            weather = response.json().get("current", {})
            a, b, c = st.columns(3)
            a.metric("Temperature", f"{weather.get('temperature_2m', '—')} °C")
            b.metric("Feels like", f"{weather.get('apparent_temperature', '—')} °C")
            c.metric("Humidity", f"{weather.get('relative_humidity_2m', '—')}%")
            d, e, f = st.columns(3)
            d.metric("Precipitation", f"{weather.get('precipitation', '—')} mm")
            e.metric("Wind speed", f"{weather.get('wind_speed_10m', '—')} km/h")
            f.metric("Weather code", str(weather.get("weather_code", "—")))
            st.caption(f"Observation time: {weather.get('time', 'not provided')}. Weather code is the provider's numeric condition code.")
        except Exception:
            st.error("Weather could not be retrieved. Check your internet connection and try again.")
    else:
        st.info("Choose a destination, then press **Get current weather** to load live conditions.")

with tab_chatbot:
    st.subheader("Tourism assistant")
    st.write("Ask about the dashboard data, forecasts, seasonality, or destination weather.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("Example: Which month has the highest average arrivals?")
    if question:
        st.session_state.chat_history.append({"role":"user", "content":question})
        q = question.lower()
        with st.chat_message("assistant"):
            if any(word in q for word in ["highest month", "best month", "peak month", "busy month"]):
                m = df.groupby("Month")["Tourist_Arrivals"].mean().idxmax()
                month_name = date(2000, int(m), 1).strftime("%B")
                answer = f"Based on the currently loaded dataset, **{month_name}** has the highest average tourist arrivals."
            elif any(word in q for word in ["lowest month", "quiet month", "least month"]):
                m = df.groupby("Month")["Tourist_Arrivals"].mean().idxmin()
                month_name = date(2000, int(m), 1).strftime("%B")
                answer = f"Based on the currently loaded dataset, **{month_name}** has the lowest average tourist arrivals."
            elif "average" in q:
                answer = f"The average monthly arrivals in the loaded dataset are **{average_monthly:,}**."
            elif any(word in q for word in ["forecast", "predict", "future"]):
                answer = "Open the **Forecast** tab, choose Random Forest or Linear Regression, and select how many months ahead you want to estimate. Forecasts are only as reliable as the historical data."
            elif "weather" in q or "temperature" in q or "rain" in q:
                answer = f"Open the **Weather** tab for {destination} and press **Get current weather** to retrieve live conditions."
            elif any(word in q for word in ["data", "dataset", "csv", "upload"]):
                answer = "Open the **Dataset** tab to inspect the loaded rows. You can upload your own CSV from the sidebar with columns Year, Month, Tourist_Arrivals."
            else:
                answer = f"I can summarize arrivals, identify peak/quiet months, explain forecasts, and direct you to live weather for {destination}. Try asking: “Which month is busiest?”"
            st.markdown(answer)
        st.session_state.chat_history.append({"role":"assistant", "content":answer})

with tab_data:
    st.subheader("Tourist-arrivals dataset")
    st.write("Expected CSV columns: `Year`, `Month`, `Tourist_Arrivals`.")
    st.dataframe(df[["Year", "Month", "Date", "Tourist_Arrivals"]], use_container_width=True, hide_index=True)
    st.download_button("Download sample dataset CSV", df[["Year", "Month", "Tourist_Arrivals"]].to_csv(index=False).encode("utf-8"),
                       file_name="sample_tourist_arrivals.csv", mime="text/csv")
    st.info("The included sample data is simulated for demonstration, not official tourism statistics.")

st.markdown("---")
st.caption("Tourism Demand Forecasting • Streamlit + Pandas + Scikit-learn + Plotly + Open-Meteo")
