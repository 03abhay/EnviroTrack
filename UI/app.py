import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import folium
from streamlit_folium import folium_static
import requests
import requests
import xml.etree.ElementTree as ET


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Environmental Monitoring Dashboard - Indian State Capitals",
    page_icon="🌍",
    layout="wide"
)



# ------------------------------------------------------------
# Basic CSS for nicer UI
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        color: #f9fafb !important;
    }

    .card {
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        background: #020617;
        border: 1px solid #1f2937;
        box-shadow: 0 10px 25px rgba(15,23,42,0.5);
    }
    .metric-card {
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
        background: #020617;
        border: 1px solid #1e293b;
    }

    .alert-bar {
        border-radius: 0.75rem;
        padding: 0.9rem 1.1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .alert-icon {
        font-size: 1.2rem;
    }
    .alert-text {
        flex: 1;
    }

    .news-card {
        border-radius: 0.75rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        background: #020617;
        border: 1px solid #1f2937;
    }
    .news-title {
        font-weight: 600;
        font-size: 0.98rem;
        color: #e5e7eb;
    }
    .news-meta {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }
    .news-desc {
        font-size: 0.9rem;
        color: #d1d5db;
    }

    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 1rem;
    }
    .footer b {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Data Loading
# ============================================================
@st.cache_data(show_spinner=False)
def load_sample_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    locations = [
        # Northern Region
        {'name': 'New Delhi (Delhi)', 'lat': 28.6139, 'lon': 77.2090},
        {'name': 'Chandigarh (Haryana & Punjab)', 'lat': 30.7333, 'lon': 76.7794},
        {'name': 'Shimla (Himachal Pradesh)', 'lat': 31.1048, 'lon': 77.1734},
        {'name': 'Srinagar (Jammu & Kashmir)', 'lat': 34.0837, 'lon': 74.7973},
        {'name': 'Leh (Ladakh)', 'lat': 34.1526, 'lon': 77.5771},
        
        # Central Region
        {'name': 'Lucknow (Uttar Pradesh)', 'lat': 26.8467, 'lon': 80.9462},
        {'name': 'Dehradun (Uttarakhand)', 'lat': 30.3165, 'lon': 78.0322},
        {'name': 'Bhopal (Madhya Pradesh)', 'lat': 23.2599, 'lon': 77.4126},
        {'name': 'Raipur (Chhattisgarh)', 'lat': 21.2514, 'lon': 81.6296},
        
        # Eastern Region
        {'name': 'Patna (Bihar)', 'lat': 25.5941, 'lon': 85.1376},
        {'name': 'Ranchi (Jharkhand)', 'lat': 23.3441, 'lon': 85.3096},
        {'name': 'Kolkata (West Bengal)', 'lat': 22.5726, 'lon': 88.3639},
        {'name': 'Gangtok (Sikkim)', 'lat': 27.3389, 'lon': 88.6065},
        
        # North Eastern Region
        {'name': 'Dispur (Assam)', 'lat': 26.1433, 'lon': 91.7898},
        {'name': 'Itanagar (Arunachal Pradesh)', 'lat': 27.0844, 'lon': 93.6053},
        {'name': 'Imphal (Manipur)', 'lat': 24.8170, 'lon': 93.9368},
        {'name': 'Shillong (Meghalaya)', 'lat': 25.5788, 'lon': 91.8933},
        {'name': 'Aizawl (Mizoram)', 'lat': 23.7307, 'lon': 92.7173},
        {'name': 'Kohima (Nagaland)', 'lat': 25.6751, 'lon': 94.1086},
        {'name': 'Agartala (Tripura)', 'lat': 23.8315, 'lon': 91.2868},
        
        # Western Region
        {'name': 'Gandhi Nagar (Gujarat)', 'lat': 23.2156, 'lon': 72.6369},
        {'name': 'Mumbai (Maharashtra)', 'lat': 19.0760, 'lon': 72.8777},
        {'name': 'Panaji (Goa)', 'lat': 15.4909, 'lon': 73.8278},
        {'name': 'Jaipur (Rajasthan)', 'lat': 26.9124, 'lon': 75.7873},
        
        # Southern Region
        {'name': 'Hyderabad (Telangana)', 'lat': 17.3850, 'lon': 78.4867},
        {'name': 'Amaravati (Andhra Pradesh)', 'lat': 16.5113, 'lon': 80.5154},
        {'name': 'Bengaluru (Karnataka)', 'lat': 12.9716, 'lon': 77.5946},
        {'name': 'Thiruvananthapuram (Kerala)', 'lat': 8.5241, 'lon': 76.9366},
        {'name': 'Chennai (Tamil Nadu)', 'lat': 13.0827, 'lon': 80.2707},
        
        # Union Territories
        {'name': 'Port Blair (Andaman & Nicobar)', 'lat': 11.6234, 'lon': 92.7265},
        {'name': 'Kavaratti (Lakshadweep)', 'lat': 10.5593, 'lon': 72.6358},
        {'name': 'Puducherry (Puducherry)', 'lat': 11.9416, 'lon': 79.8083},
        {'name': 'Daman (Daman & Diu)', 'lat': 20.3974, 'lon': 72.8328},
        {'name': 'Silvassa (Dadra & Nagar Haveli)', 'lat': 20.2766, 'lon': 73.0081}
    ]
    
    data = []
    for location in locations:
        lat = location["lat"]
        for date in dates:
            base_temp = 30 - (lat - 20) * 0.5

            if date.month in [5, 6]:  # Summer
                temp = np.random.normal(base_temp + 5, 3)
            elif date.month in [12, 1]:  # Winter
                temp = np.random.normal(base_temp - 10, 3)
            else:
                temp = np.random.normal(base_temp, 4)

            if lat > 25 and date.month in [11, 12, 1, 2]:
                aqi = np.random.normal(200, 50)
            else:
                aqi = np.random.normal(100, 30)

            if date.month in [6, 7, 8, 9]:  # Monsoon
                if 8 < lat < 20:
                    rainfall = np.random.exponential(20)
                else:
                    rainfall = np.random.exponential(15)
            else:
                rainfall = np.random.exponential(2)

            data.append({
                'date': date,
                'location': location['name'],
                'lat': location['lat'],
                'lon': location['lon'],
                'temperature': round(max(0, min(50, temp)), 2),
                'air_quality': round(max(0, min(500, aqi)), 2),
                'rainfall': round(rainfall, 2)
            })
    
    df = pd.DataFrame(data)
    return df

# ------------------------------------------------------------
# Forecast Helper
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def create_forecast(df, parameter, days=30):
    tmp = df[['date', parameter]].rename(columns={'date': 'ds', parameter: 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(tmp)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# ------------------------------------------------------------
# Helper: Extract city name for news queries
# ------------------------------------------------------------
def extract_city_keyword(location_name: str) -> str:
    # e.g. "New Delhi (Delhi)" -> "New Delhi"
    if "(" in location_name:
        return location_name.split("(")[0].strip()
    return location_name.strip()

# ------------------------------------------------------------
# Weather News API integration
# ------------------------------------------------------------


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_news(city_keyword: str):
    """
    Fetch latest weather-related news using Google News RSS.
    NO API key required.
    """
    query = f"{city_keyword} weather OR rainfall OR storm OR cyclone OR heatwave"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"articles": [], "error": f"RSS error: {response.status_code}"}

        root = ET.fromstring(response.content)

        articles = []
        for item in root.findall(".//item"):
            articles.append({
                "title": item.find("title").text,
                "description": item.find("description").text,
                "url": item.find("link").text,
                "published_at": item.find("pubDate").text,
                "source": "Google News"
            })

        return {"articles": articles[:10], "error": None}

    except Exception as e:
        return {"articles": [], "error": f"Exception: {e}"}

# ------------------------------------------------------------
# Alert / Warning Logic
# ------------------------------------------------------------
def get_alert_status(current_row, news_articles=None):
    """
    Decide alert severity + text based on:
    - Latest temp / rainfall / AQI
    - Recent severe weather news headlines for this location (if available)
    """
    temp = current_row["temperature"]
    rain = current_row["rainfall"]
    aqi = current_row["air_quality"]

    messages = []
    severity = "normal"   # normal < medium < high
    icon = "🟢"
    sev_order = ["normal", "medium", "high"]

    def bump_severity(current, new_level):
        return max(current, new_level, key=lambda s: sev_order.index(s))

    # Temperature conditions
    if temp >= 42:
        messages.append(f"Heatwave conditions detected (Temperature: {temp}°C).")
        severity = "high"
        icon = "🔥"
    elif temp >= 38:
        messages.append(f"High temperature alert (Temperature: {temp}°C).")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "☀️"
    elif temp <= 5:
        messages.append(f"Very low temperature alert (Temperature: {temp}°C).")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "❄️"

    # Rainfall / storm conditions
    if rain >= 80:
        messages.append(f"Extreme rainfall – cyclone-like / severe storm risk (Rainfall: {rain} mm).")
        severity = "high"
        icon = "🌀"
    elif rain >= 30:
        messages.append(f"Heavy rainfall – storm / flooding risk (Rainfall: {rain} mm).")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "⛈️"

    # AQI conditions
    if aqi >= 300:
        messages.append(f"Hazardous air quality (AQI: {aqi}). Limit outdoor activity.")
        severity = "high"
        if icon == "🟢":
            icon = "☠️"
    elif aqi >= 200:
        messages.append(f"Very poor air quality (AQI: {aqi}). Wear masks outdoors.")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "😷"

    # News-driven alerts
    if news_articles:
        severe_keywords_high = ["cyclone", "landslide", "red alert", "severe storm"]
        severe_keywords_med = ["flood", "flooding", "heatwave", "cold wave", "heavy rain", "orange alert"]
        flagged_article = None
        flagged_level = None

        for art in news_articles:
            text = (art["title"] + " " + (art["description"] or "")).lower()
            if any(k in text for k in severe_keywords_high):
                flagged_article = art
                flagged_level = "high"
                break
            elif any(k in text for k in severe_keywords_med) and flagged_level is None:
                flagged_article = art
                flagged_level = "medium"

        if flagged_article:
            if flagged_level == "high":
                severity = "high"
            else:
                severity = bump_severity(severity, "medium")

            if icon == "🟢":
                icon = "📰"

            messages.append(
                f"News alert: {flagged_article['title']}"
            )

    # If no issues at all
    if not messages:
        messages = ["Conditions look stable. No major weather or air quality alerts for now."]
        severity = "normal"
        icon = "🟢"

    # Decide alert bar color
    if severity == "high":
        bg = "#7f1d1d"
        border = "#fecaca"
    elif severity == "medium":
        bg = "#78350f"
        border = "#fed7aa"
    else:
        bg = "#022c22"
        border = "#bbf7d0"

    return icon, messages, bg, border

def render_alert_bar(current_row, location_name, news_articles=None):
    icon, messages, bg, border = get_alert_status(current_row, news_articles=news_articles)
    msg_html = "<br>".join(messages)

    st.markdown(
        f"""
        <div class="alert-bar" style="background:{bg}; border:1px solid {border};">
            <div class="alert-icon">{icon}</div>
            <div class="alert-text">
                <b>Alert Status for {location_name}</b><br>
                {msg_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# Main App
# ============================================================
data = load_sample_data()

# Sidebar Controls
st.sidebar.title("⚙️ Controls")

selected_location = st.sidebar.selectbox(
    "Select Location",
    sorted(data['location'].unique())
)

selected_parameter = st.sidebar.selectbox(
    "Select Parameter",
    ['temperature', 'air_quality', 'rainfall']
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [data['date'].min(), data['date'].max()]
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=60,
    value=30,
    step=7
)

st.sidebar.markdown(
    """
    ---
    **Tip:** The alert bar combines live metrics  
    + latest weather-related news headlines for the selected city.
    """
)

# Title & Subtitle
st.title("🌍 Environmental Monitoring Dashboard – Indian State Capitals")
st.markdown(
    "Real-time style insights (simulated data) for temperature, air quality, "
    "rainfall, and weather-related news across Indian state capitals and key union territories."
)

# Prep location-specific data
loc_data_all = data[data["location"] == selected_location].sort_values("date")
current_data = loc_data_all.iloc[-1]
prev_row = loc_data_all.iloc[-2]

# Fetch news for selected city
city_keyword = extract_city_keyword(selected_location)
news_data = fetch_weather_news(city_keyword)
weather_news = news_data["articles"]
news_error = news_data["error"]

# ============================================================
# ALERT BAR (Top) – now also uses news
# ============================================================
render_alert_bar(current_data, selected_location, news_articles=weather_news)

# ============================================================
# Layout Tabs (added Weather News tab)
# ============================================================
tab_overview, tab_trend, tab_data, tab_news = st.tabs(
    ["🌐 Overview", "📈 Trends & Forecast", "📊 Raw Data", "📰 Weather News"]
)

# ------------------------------------------------------------
# OVERVIEW TAB
# ------------------------------------------------------------
with tab_overview:
    col_map, col_stats = st.columns([2.2, 1])

    with col_map:
        st.subheader("Geographic Visualization")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        m = folium.Map(
            location=[20.5937, 78.9629],
            zoom_start=4,
            tiles="CartoDB positron"
        )

        latest_per_loc = data.sort_values("date").groupby("location").tail(1)
        for _, row in latest_per_loc.iterrows():
            popup_text = (
                f"<b>{row['location']}</b><br>"
                f"Temp: {row['temperature']}°C<br>"
                f"AQI: {row['air_quality']}<br>"
                f"Rainfall: {row['rainfall']} mm"
            )
            folium.Marker(
                [row['lat'], row['lon']],
                popup=popup_text,
                tooltip=row['location']
            ).add_to(m)

        folium_static(m, width=700, height=400)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_stats:
        st.subheader("Current Snapshot")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Temperature")
            st.metric(
                label="°C",
                value=f"{current_data['temperature']:.1f}",
                delta=f"{current_data['temperature'] - prev_row['temperature']:.1f}"
            )

        with col_b:
            st.markdown("##### Air Quality Index")
            st.metric(
                label="AQI",
                value=f"{current_data['air_quality']:.1f}",
                delta=f"{current_data['air_quality'] - prev_row['air_quality']:.1f}"
            )

        st.markdown("---")
        st.markdown("##### Rainfall")
        st.metric(
            label="mm",
            value=f"{current_data['rainfall']:.1f}",
            delta=f"{current_data['rainfall'] - prev_row['rainfall']:.1f}"
        )

        st.markdown(
            """
            <small>
            *Metrics compare today's values with the previous recorded day for the same city.*
            </small>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TREND & FORECAST TAB
# ------------------------------------------------------------
with tab_trend:
    st.subheader(f"{selected_parameter.title()} – Trend & Forecast for {selected_location}")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    filtered_data = data[
        (data['location'] == selected_location) &
        (data['date'] >= pd.Timestamp(date_range[0])) &
        (data['date'] <= pd.Timestamp(date_range[1]))
    ]

    fig = px.line(
        filtered_data,
        x='date',
        y=selected_parameter,
        title=f"{selected_parameter.title()} Trend ({selected_location})",
        labels={'date': 'Date', selected_parameter: selected_parameter.title()}
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Forecast")
    st.write(
        f"Generate a {forecast_days}-day forecast using Prophet based on the full history of "
        f"{selected_parameter.replace('_', ' ').title()} for **{selected_location}**."
    )

    if st.button("🔮 Generate Forecast"):
        with st.spinner("Training Prophet model and generating forecast..."):
            forecast = create_forecast(loc_data_all, selected_parameter, days=forecast_days)

            fig_fc = go.Figure()

            fig_fc.add_trace(go.Scatter(
                x=loc_data_all['date'],
                y=loc_data_all[selected_parameter],
                name='Actual',
                mode='lines'
            ))

            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast',
                mode='lines',
                line=dict(dash='dash')
            ))

            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                name='Upper Bound',
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))

            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                name='Lower Bound',
                mode='lines',
                fill='tonexty',
                showlegend=True
            ))

            fig_fc.update_layout(
                title=f"{selected_parameter.title()} Forecast – {selected_location}",
                xaxis_title="Date",
                yaxis_title=selected_parameter.title(),
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# RAW DATA TAB
# ------------------------------------------------------------
with tab_data:
    st.subheader(f"Raw Data – {selected_location}")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.dataframe(
        loc_data_all.sort_values("date", ascending=False),
        use_container_width=True,
        height=400
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# WEATHER NEWS TAB
# ------------------------------------------------------------
with tab_news:
    st.subheader(f"Latest Weather News – {city_keyword}")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if news_error:
        st.warning(news_error)
    elif not weather_news:
        st.info("No recent weather-related news articles found for this region.")
    else:
        for art in weather_news:
            published = art["published_at"]
            src = art["source"]

            st.markdown(
                f"""
                <div class="news-card">
                    <div class="news-title">{art['title']}</div>
                    <div class="news-meta">{src} · {published}</div>
                    <div class="news-desc">{art['description'] or ""}</div>
                    <div style="margin-top:0.4rem;">
                        <a href="{art['url']}" target="_blank">Read full article ↗</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER (Same style as InvestIQ, adapted)
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <b>Environmental Monitoring Dashboard – Indian State Capitals</b><br>
        Simulated environmental metrics combined with third-party news data  
        for educational and demonstration purposes only.<br>
        Do not use for real-world emergency or policy decisions.<br><br>
        Built by <b>Abhay Singh</b> · © 2025 Abhay Singh. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
