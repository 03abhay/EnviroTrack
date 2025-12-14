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
import xml.etree.ElementTree as ET


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Environmental Monitoring Dashboard - Indian State Capitals",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark theme
st.markdown("""
    <script>
    window.parent.document.documentElement.setAttribute('data-theme', 'dark');
    </script>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Enhanced CSS with animations
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Force dark theme */
    :root {
        color-scheme: dark;
    }
    
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

    /* Alert bar with pulse animation for high severity */
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
        transition: all 0.3s ease;
    }
    
    .alert-bar.high-severity {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        }
        50% {
            box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
        }
    }
    
    .alert-icon {
        font-size: 1.2rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-5px);
        }
    }
    
    .alert-text {
        flex: 1;
    }

    /* Legend card styles */
    .legend-card {
        border-radius: 0.75rem;
        padding: 1rem;
        background: #020617;
        border: 1px solid #1f2937;
        margin-bottom: 1rem;
    }
    
    .legend-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f9fafb;
        margin-bottom: 0.75rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 2px solid #374151;
    }
    
    .legend-range {
        color: #9ca3af;
        flex: 1;
    }

    .news-card {
        border-radius: 0.75rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        background: #020617;
        border: 1px solid #1f2937;
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        transform: translateY(-2px);
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
    
    /* Streamlit element overrides for dark theme */
    .stSelectbox > div > div {
        background-color: #1e293b;
        color: #e5e7eb;
    }
    
    .stDateInput > div > div {
        background-color: #1e293b;
        color: #e5e7eb;
    }
    
    /* Glowing effect for interactive elements */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
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
    # Generate data for last 60 days from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    locations = [
        # Jammu & Kashmir (including PoK - Muzaffarabad as integral part)
        {'name': 'Srinagar (Jammu & Kashmir)', 'lat': 34.0837, 'lon': 74.7973},
        {'name': 'Jammu (Jammu & Kashmir)', 'lat': 32.7266, 'lon': 74.8570},
        {'name': 'Anantnag (Jammu & Kashmir)', 'lat': 33.7310, 'lon': 75.1484},
        {'name': 'Muzaffarabad (Jammu & Kashmir - PoK)', 'lat': 34.3700, 'lon': 73.4711},
        
        # Ladakh (including Aksai Chin areas)
        {'name': 'Leh (Ladakh)', 'lat': 34.1526, 'lon': 77.5771},
        {'name': 'Kargil (Ladakh)', 'lat': 34.5539, 'lon': 76.1313},
        {'name': 'Aksai Chin Region (Ladakh)', 'lat': 35.3000, 'lon': 79.0000},
        
        # Himachal Pradesh
        {'name': 'Shimla (Himachal Pradesh)', 'lat': 31.1048, 'lon': 77.1734},
        {'name': 'Dharamshala (Himachal Pradesh)', 'lat': 32.2190, 'lon': 76.3234},
        {'name': 'Manali (Himachal Pradesh)', 'lat': 32.2396, 'lon': 77.1887},
        
        # Punjab
        {'name': 'Chandigarh (Punjab & Haryana)', 'lat': 30.7333, 'lon': 76.7794},
        {'name': 'Amritsar (Punjab)', 'lat': 31.6340, 'lon': 74.8723},
        {'name': 'Ludhiana (Punjab)', 'lat': 30.9010, 'lon': 75.8573},
        {'name': 'Patiala (Punjab)', 'lat': 30.3398, 'lon': 76.3869},
        
        # Haryana
        {'name': 'Gurugram (Haryana)', 'lat': 28.4595, 'lon': 77.0266},
        {'name': 'Faridabad (Haryana)', 'lat': 28.4089, 'lon': 77.3178},
        {'name': 'Panipat (Haryana)', 'lat': 29.3909, 'lon': 76.9635},
        
        # Delhi
        {'name': 'New Delhi (Delhi)', 'lat': 28.6139, 'lon': 77.2090},
        {'name': 'Dwarka (Delhi)', 'lat': 28.5921, 'lon': 77.0460},
        {'name': 'Rohini (Delhi)', 'lat': 28.7499, 'lon': 77.0672},
        
        # Uttarakhand
        {'name': 'Dehradun (Uttarakhand)', 'lat': 30.3165, 'lon': 78.0322},
        {'name': 'Haridwar (Uttarakhand)', 'lat': 29.9457, 'lon': 78.1642},
        {'name': 'Nainital (Uttarakhand)', 'lat': 29.3803, 'lon': 79.4636},
        
        # Uttar Pradesh
        {'name': 'Lucknow (Uttar Pradesh)', 'lat': 26.8467, 'lon': 80.9462},
        {'name': 'Agra (Uttar Pradesh)', 'lat': 27.1767, 'lon': 78.0081},
        {'name': 'Varanasi (Uttar Pradesh)', 'lat': 25.3176, 'lon': 82.9739},
        {'name': 'Kanpur (Uttar Pradesh)', 'lat': 26.4499, 'lon': 80.3319},
        
        # Rajasthan
        {'name': 'Jaipur (Rajasthan)', 'lat': 26.9124, 'lon': 75.7873},
        {'name': 'Jodhpur (Rajasthan)', 'lat': 26.2389, 'lon': 73.0243},
        {'name': 'Udaipur (Rajasthan)', 'lat': 24.5854, 'lon': 73.7125},
        {'name': 'Jaisalmer (Rajasthan)', 'lat': 26.9157, 'lon': 70.9083},
        
        # Gujarat
        {'name': 'Gandhinagar (Gujarat)', 'lat': 23.2156, 'lon': 72.6369},
        {'name': 'Ahmedabad (Gujarat)', 'lat': 23.0225, 'lon': 72.5714},
        {'name': 'Surat (Gujarat)', 'lat': 21.1702, 'lon': 72.8311},
        {'name': 'Vadodara (Gujarat)', 'lat': 22.3072, 'lon': 73.1812},
        
        # Madhya Pradesh
        {'name': 'Bhopal (Madhya Pradesh)', 'lat': 23.2599, 'lon': 77.4126},
        {'name': 'Indore (Madhya Pradesh)', 'lat': 22.7196, 'lon': 75.8577},
        {'name': 'Gwalior (Madhya Pradesh)', 'lat': 26.2183, 'lon': 78.1828},
        {'name': 'Ujjain (Madhya Pradesh)', 'lat': 23.1765, 'lon': 75.7885},
        
        # Chhattisgarh
        {'name': 'Raipur (Chhattisgarh)', 'lat': 21.2514, 'lon': 81.6296},
        {'name': 'Bhilai (Chhattisgarh)', 'lat': 21.2167, 'lon': 81.3833},
        {'name': 'Bilaspur (Chhattisgarh)', 'lat': 22.0797, 'lon': 82.1409},
        
        # Maharashtra
        {'name': 'Mumbai (Maharashtra)', 'lat': 19.0760, 'lon': 72.8777},
        {'name': 'Pune (Maharashtra)', 'lat': 18.5204, 'lon': 73.8567},
        {'name': 'Nagpur (Maharashtra)', 'lat': 21.1458, 'lon': 79.0882},
        {'name': 'Nashik (Maharashtra)', 'lat': 19.9975, 'lon': 73.7898},
        
        # Goa
        {'name': 'Panaji (Goa)', 'lat': 15.4909, 'lon': 73.8278},
        {'name': 'Vasco da Gama (Goa)', 'lat': 15.3989, 'lon': 73.8150},
        {'name': 'Margao (Goa)', 'lat': 15.2708, 'lon': 73.9528},
        
        # Karnataka
        {'name': 'Bengaluru (Karnataka)', 'lat': 12.9716, 'lon': 77.5946},
        {'name': 'Mysuru (Karnataka)', 'lat': 12.2958, 'lon': 76.6394},
        {'name': 'Mangaluru (Karnataka)', 'lat': 12.9141, 'lon': 74.8560},
        {'name': 'Hubballi (Karnataka)', 'lat': 15.3647, 'lon': 75.1240},
        
        # Telangana
        {'name': 'Hyderabad (Telangana)', 'lat': 17.3850, 'lon': 78.4867},
        {'name': 'Warangal (Telangana)', 'lat': 17.9689, 'lon': 79.5941},
        {'name': 'Nizamabad (Telangana)', 'lat': 18.6725, 'lon': 78.0941},
        
        # Andhra Pradesh
        {'name': 'Amaravati (Andhra Pradesh)', 'lat': 16.5113, 'lon': 80.5154},
        {'name': 'Visakhapatnam (Andhra Pradesh)', 'lat': 17.6868, 'lon': 83.2185},
        {'name': 'Vijayawada (Andhra Pradesh)', 'lat': 16.5062, 'lon': 80.6480},
        {'name': 'Tirupati (Andhra Pradesh)', 'lat': 13.6288, 'lon': 79.4192},
        
        # Tamil Nadu
        {'name': 'Chennai (Tamil Nadu)', 'lat': 13.0827, 'lon': 80.2707},
        {'name': 'Coimbatore (Tamil Nadu)', 'lat': 11.0168, 'lon': 76.9558},
        {'name': 'Madurai (Tamil Nadu)', 'lat': 9.9252, 'lon': 78.1198},
        {'name': 'Tiruchirappalli (Tamil Nadu)', 'lat': 10.7905, 'lon': 78.7047},
        
        # Kerala
        {'name': 'Thiruvananthapuram (Kerala)', 'lat': 8.5241, 'lon': 76.9366},
        {'name': 'Kochi (Kerala)', 'lat': 9.9312, 'lon': 76.2673},
        {'name': 'Kozhikode (Kerala)', 'lat': 11.2588, 'lon': 75.7804},
        {'name': 'Thrissur (Kerala)', 'lat': 10.5276, 'lon': 76.2144},
        
        # Bihar
        {'name': 'Patna (Bihar)', 'lat': 25.5941, 'lon': 85.1376},
        {'name': 'Gaya (Bihar)', 'lat': 24.7955, 'lon': 84.9994},
        {'name': 'Bhagalpur (Bihar)', 'lat': 25.2425, 'lon': 86.9842},
        
        # Jharkhand
        {'name': 'Ranchi (Jharkhand)', 'lat': 23.3441, 'lon': 85.3096},
        {'name': 'Jamshedpur (Jharkhand)', 'lat': 22.8046, 'lon': 86.2029},
        {'name': 'Dhanbad (Jharkhand)', 'lat': 23.7957, 'lon': 86.4304},
        
        # Odisha
        {'name': 'Bhubaneswar (Odisha)', 'lat': 20.2961, 'lon': 85.8245},
        {'name': 'Cuttack (Odisha)', 'lat': 20.4625, 'lon': 85.8830},
        {'name': 'Puri (Odisha)', 'lat': 19.8135, 'lon': 85.8312},
        
        # West Bengal
        {'name': 'Kolkata (West Bengal)', 'lat': 22.5726, 'lon': 88.3639},
        {'name': 'Howrah (West Bengal)', 'lat': 22.5958, 'lon': 88.2636},
        {'name': 'Siliguri (West Bengal)', 'lat': 26.7271, 'lon': 88.3953},
        {'name': 'Darjeeling (West Bengal)', 'lat': 27.0360, 'lon': 88.2627},
        
        # Sikkim
        {'name': 'Gangtok (Sikkim)', 'lat': 27.3389, 'lon': 88.6065},
        {'name': 'Namchi (Sikkim)', 'lat': 27.1667, 'lon': 88.3667},
        {'name': 'Pelling (Sikkim)', 'lat': 27.2871, 'lon': 88.2150},
        
        # Assam
        {'name': 'Dispur (Assam)', 'lat': 26.1433, 'lon': 91.7898},
        {'name': 'Guwahati (Assam)', 'lat': 26.1445, 'lon': 91.7362},
        {'name': 'Silchar (Assam)', 'lat': 24.8333, 'lon': 92.7789},
        
        # Arunachal Pradesh
        {'name': 'Itanagar (Arunachal Pradesh)', 'lat': 27.0844, 'lon': 93.6053},
        {'name': 'Tawang (Arunachal Pradesh)', 'lat': 27.5860, 'lon': 91.8590},
        {'name': 'Ziro (Arunachal Pradesh)', 'lat': 27.5450, 'lon': 93.8317},
        
        # Nagaland
        {'name': 'Kohima (Nagaland)', 'lat': 25.6751, 'lon': 94.1086},
        {'name': 'Dimapur (Nagaland)', 'lat': 25.9040, 'lon': 93.7267},
        {'name': 'Mokokchung (Nagaland)', 'lat': 26.3217, 'lon': 94.5203},
        
        # Manipur
        {'name': 'Imphal (Manipur)', 'lat': 24.8170, 'lon': 93.9368},
        {'name': 'Thoubal (Manipur)', 'lat': 24.6333, 'lon': 93.9833},
        {'name': 'Bishnupur (Manipur)', 'lat': 24.6000, 'lon': 93.7667},
        
        # Mizoram
        {'name': 'Aizawl (Mizoram)', 'lat': 23.7307, 'lon': 92.7173},
        {'name': 'Lunglei (Mizoram)', 'lat': 22.8900, 'lon': 92.7347},
        {'name': 'Champhai (Mizoram)', 'lat': 23.4697, 'lon': 93.3269},
        
        # Tripura
        {'name': 'Agartala (Tripura)', 'lat': 23.8315, 'lon': 91.2868},
        {'name': 'Udaipur (Tripura)', 'lat': 23.5333, 'lon': 91.4833},
        {'name': 'Dharmanagar (Tripura)', 'lat': 24.3667, 'lon': 92.1667},
        
        # Meghalaya
        {'name': 'Shillong (Meghalaya)', 'lat': 25.5788, 'lon': 91.8933},
        {'name': 'Tura (Meghalaya)', 'lat': 25.5138, 'lon': 90.2034},
        {'name': 'Cherrapunji (Meghalaya)', 'lat': 25.2697, 'lon': 91.7320},
        
        # Andaman & Nicobar
        {'name': 'Port Blair (Andaman & Nicobar)', 'lat': 11.6234, 'lon': 92.7265},
        {'name': 'Diglipur (Andaman & Nicobar)', 'lat': 13.2667, 'lon': 93.0000},
        {'name': 'Car Nicobar (Andaman & Nicobar)', 'lat': 9.1528, 'lon': 92.8194},
        
        # Puducherry
        {'name': 'Puducherry (Puducherry)', 'lat': 11.9416, 'lon': 79.8083},
        {'name': 'Karaikal (Puducherry)', 'lat': 10.9254, 'lon': 79.8380},
        {'name': 'Mahe (Puducherry)', 'lat': 11.7009, 'lon': 75.5340},
        
        # Lakshadweep
        {'name': 'Kavaratti (Lakshadweep)', 'lat': 10.5593, 'lon': 72.6358},
        {'name': 'Agatti (Lakshadweep)', 'lat': 10.8482, 'lon': 72.1920},
        {'name': 'Minicoy (Lakshadweep)', 'lat': 8.2833, 'lon': 73.0500},
        
        # Daman & Diu
        {'name': 'Daman (Daman & Diu)', 'lat': 20.3974, 'lon': 72.8328},
        {'name': 'Diu (Daman & Diu)', 'lat': 20.7144, 'lon': 70.9882},
        
        # Dadra & Nagar Haveli
        {'name': 'Silvassa (Dadra & Nagar Haveli)', 'lat': 20.2766, 'lon': 73.0081},
        {'name': 'Dadra (Dadra & Nagar Haveli)', 'lat': 20.2700, 'lon': 73.0150}
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
# Severity Assessment Function
# ------------------------------------------------------------
def assess_severity(temp, rain, aqi):
    """Returns 'normal', 'warning', or 'severe'"""
    if temp >= 42 or rain >= 80 or aqi >= 300:
        return 'severe'
    elif temp >= 38 or temp <= 5 or rain >= 30 or aqi >= 200:
        return 'warning'
    else:
        return 'normal'

def get_marker_color(severity):
    """Returns color code for map markers"""
    if severity == 'severe':
        return 'red'
    elif severity == 'warning':
        return 'orange'
    else:
        return 'green'

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
    if "(" in location_name:
        return location_name.split("(")[0].strip()
    return location_name.strip()

# ------------------------------------------------------------
# Weather News API integration
# ------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_news(city_keyword: str):
    """Fetch latest weather-related news using Google News RSS."""
    query = f"{city_keyword} weather OR rainfall OR storm OR cyclone OR heatwave"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"articles": [], "error": f"RSS error: {response.status_code}"}

        root = ET.fromstring(response.content)

        articles = []
        for item in root.findall(".//item"):
            title_text = item.find("title").text if item.find("title") is not None else ""
            desc_text = item.find("description").text if item.find("description") is not None else ""
            url_text = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            
            articles.append({
                "title": title_text,
                "description": desc_text,
                "url": url_text,
                "published_at": pub_date,
                "source": "Google News",
                "pub_timestamp": pd.to_datetime(pub_date, errors='coerce')
            })

        # Sort by publication date (most recent first)
        articles_df = pd.DataFrame(articles)
        if not articles_df.empty and 'pub_timestamp' in articles_df.columns:
            articles_df = articles_df.sort_values('pub_timestamp', ascending=False)
            articles = articles_df.to_dict('records')

        return {"articles": articles[:15], "error": None}  # Return top 15 most recent

    except Exception as e:
        return {"articles": [], "error": f"Exception: {e}"}

# ------------------------------------------------------------
# Alert / Warning Logic (Past Week Only)
# ------------------------------------------------------------
def get_alert_status(current_row, location_data_week, news_articles=None):
    """
    Decide alert severity + text based on:
    - Current metrics
    - Past week's weather patterns
    - Recent news (last 7 days only)
    """
    temp = current_row["temperature"]
    rain = current_row["rainfall"]
    aqi = current_row["air_quality"]

    messages = []
    severity = "normal"
    icon = "🟢"
    sev_order = ["normal", "medium", "high"]

    def bump_severity(current, new_level):
        return max(current, new_level, key=lambda s: sev_order.index(s))

    # Analyze past week patterns
    week_avg_temp = location_data_week['temperature'].mean()
    week_max_temp = location_data_week['temperature'].max()
    week_total_rain = location_data_week['rainfall'].sum()
    week_avg_aqi = location_data_week['air_quality'].mean()
    
    # Current Temperature conditions
    if temp >= 42:
        messages.append(f"🔥 Heatwave conditions detected (Current: {temp}°C, Week Avg: {week_avg_temp:.1f}°C).")
        severity = "high"
        icon = "🔥"
    elif temp >= 38:
        messages.append(f"☀️ High temperature alert (Current: {temp}°C).")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "☀️"
    elif temp <= 5:
        messages.append(f"❄️ Very low temperature alert (Current: {temp}°C).")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "❄️"
    
    # Check for unusual temperature changes in past week
    if week_max_temp - temp > 10:
        messages.append(f"⚠️ Significant temperature drop from week's peak ({week_max_temp:.1f}°C to {temp:.1f}°C).")
        severity = bump_severity(severity, "medium")

    # Current Rainfall / storm conditions
    if rain >= 80:
        messages.append(f"🌀 Extreme rainfall – cyclone-like / severe storm risk (Current: {rain} mm, Week Total: {week_total_rain:.1f} mm).")
        severity = "high"
        icon = "🌀"
    elif rain >= 30:
        messages.append(f"⛈️ Heavy rainfall – storm / flooding risk (Current: {rain} mm).")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "⛈️"
    
    # Check weekly rainfall accumulation
    if week_total_rain >= 200:
        messages.append(f"💧 Very high weekly rainfall accumulation ({week_total_rain:.1f} mm) - flood risk remains elevated.")
        severity = bump_severity(severity, "medium")

    # Current AQI conditions
    if aqi >= 300:
        messages.append(f"☠️ Hazardous air quality (Current AQI: {aqi}, Week Avg: {week_avg_aqi:.0f}). Limit outdoor activity.")
        severity = "high"
        if icon == "🟢":
            icon = "☠️"
    elif aqi >= 200:
        messages.append(f"😷 Very poor air quality (Current AQI: {aqi}). Wear masks outdoors.")
        severity = bump_severity(severity, "medium")
        if icon == "🟢":
            icon = "😷"

    # News-driven alerts (ONLY from past 7 days)
    if news_articles:
        from datetime import datetime, timezone
        import dateutil.parser
        
        # Filter news from past 7 days only
        recent_articles = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
        
        for art in news_articles:
            try:
                pub_timestamp = art.get('pub_timestamp')
                if pub_timestamp and pd.notna(pub_timestamp):
                    # Make timestamp timezone-aware if it isn't
                    if pub_timestamp.tzinfo is None:
                        pub_timestamp = pub_timestamp.tz_localize('UTC')
                    
                    if pub_timestamp >= cutoff_date:
                        recent_articles.append(art)
            except:
                # If parsing fails, skip this article
                continue
        
        # Only check recent articles (past week)
        severe_keywords_high = ["cyclone", "landslide", "red alert", "severe storm", "emergency", "disaster"]
        severe_keywords_med = ["flood", "flooding", "heatwave", "cold wave", "heavy rain", "orange alert", "yellow alert"]
        flagged_article = None
        flagged_level = None

        for art in recent_articles:
            title_clean = art.get("title", "")
            desc_clean = art.get("description", "")
            text = (title_clean + " " + desc_clean).lower()
            
            if any(k in text for k in severe_keywords_high):
                flagged_article = art
                flagged_level = "high"
                break
            elif any(k in text for k in severe_keywords_med) and flagged_level is None:
                flagged_article = art
                flagged_level = "medium"

        if flagged_article:
            import re
            import html
            
            # Clean the title
            title_clean = html.unescape(re.sub('<[^<]+?>', '', flagged_article.get('title', '')))
            
            if flagged_level == "high":
                severity = "high"
                messages.append(f"🚨 RECENT NEWS ALERT: {title_clean}")
            else:
                severity = bump_severity(severity, "medium")
                messages.append(f"📰 Recent weather advisory: {title_clean}")

            if icon == "🟢":
                icon = "📰"

    # If no issues at all
    if not messages:
        messages = [f"✅ Conditions look stable. No major alerts in the past week. (Avg Temp: {week_avg_temp:.1f}°C, Total Rain: {week_total_rain:.1f}mm, Avg AQI: {week_avg_aqi:.0f})"]
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

    return icon, messages, bg, border, severity

def render_alert_bar(current_row, location_name, location_data_week, news_articles=None):
    icon, messages, bg, border, severity = get_alert_status(current_row, location_data_week, news_articles=news_articles)
    msg_html = "<br>".join(messages)
    
    severity_class = "high-severity" if severity == "high" else ""

    st.markdown(
        f"""
        <div class="alert-bar {severity_class}" style="background:{bg}; border:1px solid {border};">
            <div class="alert-icon">{icon}</div>
            <div class="alert-text">
                <b>Alert Status for {location_name} (Past 7 Days Analysis)</b><br>
                {msg_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# Legend Components
# ------------------------------------------------------------
def render_aqi_legend():
    st.markdown(
        """
        <div class="legend-card">
            <div class="legend-title">📊 Air Quality Index (AQI) Scale</div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #10b981;"></div>
                <span class="legend-range"><b>0-50:</b> Good</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #fbbf24;"></div>
                <span class="legend-range"><b>51-100:</b> Moderate</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f97316;"></div>
                <span class="legend-range"><b>101-200:</b> Unhealthy for Sensitive Groups</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ef4444;"></div>
                <span class="legend-range"><b>201-300:</b> Very Unhealthy</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #991b1b;"></div>
                <span class="legend-range"><b>300+:</b> Hazardous</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_temp_legend():
    st.markdown(
        """
        <div class="legend-card">
            <div class="legend-title">🌡️ Temperature Thresholds</div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #3b82f6;"></div>
                <span class="legend-range"><b>&lt;5°C:</b> Cold Wave Alert</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #10b981;"></div>
                <span class="legend-range"><b>5-38°C:</b> Normal Range</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f97316;"></div>
                <span class="legend-range"><b>38-42°C:</b> High Temperature</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ef4444;"></div>
                <span class="legend-range"><b>42°C+:</b> Heatwave Conditions</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_map_legend():
    st.markdown(
        """
        <div class="legend-card">
            <div class="legend-title">🗺️ Map Marker Legend</div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #10b981;"></div>
                <span class="legend-range"><b>Green:</b> Normal conditions</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f97316;"></div>
                <span class="legend-range"><b>Orange:</b> Warning level</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ef4444;"></div>
                <span class="legend-range"><b>Red:</b> Severe conditions</span>
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

# Set default date range to last 60 days
data_end = data['date'].max()
data_start = data['date'].min()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [data_start.date(), data_end.date()],
    min_value=data_start.date(),
    max_value=data_end.date()
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=60,
    value=30,
    step=7
)

st.sidebar.markdown("---")
render_map_legend()

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

# Get past week data for alert analysis
week_ago = loc_data_all['date'].max() - timedelta(days=7)
loc_data_week = loc_data_all[loc_data_all['date'] >= week_ago]

# Fetch news for selected city
city_keyword = extract_city_keyword(selected_location)
news_data = fetch_weather_news(city_keyword)
weather_news = news_data["articles"]
news_error = news_data["error"]

# ============================================================
# ALERT BAR (Top) – now uses past week data + recent news only
# ============================================================
render_alert_bar(current_data, selected_location, loc_data_week, news_articles=weather_news)

# ============================================================
# Layout Tabs
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
        
        # Map type selector
        map_type = st.radio(
            "Map Style",
            ["Satellite", "Street View"],
            horizontal=True,
            key="map_type_selector"
        )
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Create map based on selection
        if map_type == "Satellite":
            # Satellite view using ESRI World Imagery
            m = folium.Map(
                location=[20.5937, 78.9629],
                zoom_start=4,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )
        else:
            # Light street view
            m = folium.Map(
                location=[20.5937, 78.9629],
                zoom_start=4,
                tiles="OpenStreetMap"
            )

        latest_per_loc = data.sort_values("date").groupby("location").tail(1)
        for _, row in latest_per_loc.iterrows():
            # Assess severity for color coding
            severity = assess_severity(row['temperature'], row['rainfall'], row['air_quality'])
            marker_color = get_marker_color(severity)
            
            popup_text = (
                f"<b>{row['location']}</b><br>"
                f"Status: <b style='color:{marker_color}'>{severity.upper()}</b><br>"
                f"Temp: {row['temperature']}°C<br>"
                f"AQI: {row['air_quality']}<br>"
                f"Rainfall: {row['rainfall']} mm"
            )
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                popup=popup_text,
                tooltip=row['location'],
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.7,
                weight=2
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
        
        # Add legends
        render_aqi_legend()
        render_temp_legend()

# ------------------------------------------------------------
# TREND & FORECAST TAB
# ------------------------------------------------------------
with tab_trend:
    st.subheader(f"{selected_parameter.title()} – Trend & Forecast for {selected_location}")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Handle date_range properly - ensure it's a tuple/list with 2 dates
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        # Fallback to all data if date_range is invalid
        start_date = data['date'].min()
        end_date = data['date'].max()

    filtered_data = data[
        (data['location'] == selected_location) &
        (data['date'] >= start_date) &
        (data['date'] <= end_date)
    ]

    if filtered_data.empty:
        st.warning("⚠️ No data available for the selected date range. Please adjust your selection.")
    else:
        fig = px.line(
            filtered_data,
            x='date',
            y=selected_parameter,
            title=f"{selected_parameter.title()} Trend ({selected_location})",
            labels={'date': 'Date', selected_parameter: selected_parameter.title()}
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            template='plotly_dark',
            hovermode='x unified'
        )
        fig.update_traces(line=dict(width=3))
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
                mode='lines',
                line=dict(width=3, color='#3b82f6')
            ))

            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast',
                mode='lines',
                line=dict(dash='dash', width=3, color='#10b981')
            ))

            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                name='Upper Bound',
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                name='Confidence Interval',
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))

            fig_fc.update_layout(
                title=f"{selected_parameter.title()} Forecast – {selected_location}",
                xaxis_title="Date",
                yaxis_title=selected_parameter.title(),
                margin=dict(l=10, r=10, t=40, b=10),
                template='plotly_dark',
                hovermode='x unified'
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
    st.markdown(
        """
        <div style="background: #1e293b; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <small>📅 Showing most recent news articles first (sorted by publication date)</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if news_error:
        st.warning(f"⚠️ {news_error}")
    elif not weather_news:
        st.info(f"🔍 No recent weather-related news articles found for {city_keyword}.")
    else:
        import re
        import html
        
        for idx, art in enumerate(weather_news, 1):
            published = art["published_at"]
            src = art["source"]
            title = art.get('title', 'No title')
            description = art.get('description', '')
            url = art.get('url', '#')
            
            # Clean title - remove HTML tags and decode entities
            title_clean = html.unescape(re.sub('<[^<]+?>', '', title))
            
            # Clean description - remove all HTML/CSS and decode entities
            if description:
                # Remove all HTML tags
                description_clean = re.sub('<[^<]+?>', '', description)
                # Decode HTML entities
                description_clean = html.unescape(description_clean)
                # Remove extra whitespace
                description_clean = ' '.join(description_clean.split())
                # Limit length
                if len(description_clean) > 250:
                    description_clean = description_clean[:250] + "..."
            else:
                description_clean = "Click to read the full article"
            
            # Add visual timestamp indicator
            if idx == 1:
                time_badge = '<span style="background: #3b82f6; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 600;">#1 Most Recent</span>'
            else:
                time_badge = f'<span style="background: #475569; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">#{idx}</span>'

            # Use st.markdown for the card
            st.markdown(
                f"""
                <div style="border-radius: 0.75rem; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem; 
                     background: #020617; border: 1px solid #1f2937; transition: all 0.3s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="font-size: 0.78rem; color: #9ca3af;">{src} · {published}</div>
                        {time_badge}
                    </div>
                    <div style="font-weight: 600; font-size: 0.98rem; color: #e5e7eb; margin-bottom: 0.4rem;">
                        {title_clean}
                    </div>
                    <div style="font-size: 0.9rem; color: #d1d5db; margin-bottom: 0.5rem;">
                        {description_clean}
                    </div>
                    <a href="{url}" target="_blank" style="color: #3b82f6; text-decoration: none; font-weight: 500; font-size: 0.9rem;">
                        📰 Read full article ↗
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
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
