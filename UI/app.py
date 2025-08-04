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
import json

# Set page configuration
st.set_page_config(
    page_title="Environmental Monitoring Dashboard - Indian State Capitals",
    page_icon="🌍",
    layout="wide"
)

# Function to load sample data (replace with actual data source in production)
@st.cache_data
def load_sample_data():
    # Generate sample data for demonstration
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
        for date in dates:
            # Adjust temperature based on region and season
            lat = location['lat']
            
            # Base temperature adjustments by latitude (generally hotter in south)
            base_temp = 30 - (lat - 20) * 0.5
            
            # Seasonal adjustments
            if date.month in [5, 6]:  # Summer months
                temp = np.random.normal(base_temp + 5, 3)
            elif date.month in [12, 1]:  # Winter months
                temp = np.random.normal(base_temp - 10, 3)
            else:  # Other months
                temp = np.random.normal(base_temp, 4)
            
            # AQI adjustments (higher in northern cities during winter)
            if lat > 25 and date.month in [11, 12, 1, 2]:
                aqi = np.random.normal(200, 50)
            else:
                aqi = np.random.normal(100, 30)
            
            # Rainfall adjustments based on monsoon and region
            if date.month in [6, 7, 8, 9]:  # Monsoon season
                if 8 < lat < 20:  # Southern region gets more rainfall
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
                'temperature': round(max(0, min(50, temp)), 2),  # Clamp between 0-50°C
                'air_quality': round(max(0, min(500, aqi)), 2),  # Clamp between 0-500
                'rainfall': round(rainfall, 2)
            })
    
    return pd.DataFrame(data)

# Function to create forecast
def create_forecast(data, parameter, days=30):
    df = data[['date', parameter]].rename(columns={'date': 'ds', parameter: 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# Load data
data = load_sample_data()

# Sidebar
st.sidebar.title("Controls")
selected_location = st.sidebar.selectbox(
    "Select Location",
    sorted(data['location'].unique())  # Sort alphabetically for easier navigation
)

selected_parameter = st.sidebar.selectbox(
    "Select Parameter",
    ['temperature', 'air_quality', 'rainfall']
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [data['date'].min(), data['date'].max()]
)

# Main content
st.title("Environmental Monitoring Dashboard - Indian State Capitals")

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Geographic Visualization")
    
    # Create map centered on India
    m = folium.Map(
        location=[20.5937, 78.9629],  # Center of India
        zoom_start=4
    )
    
    # Add markers for all locations
    for loc in data['location'].unique():
        loc_data = data[data['location'] == loc].iloc[0]
        folium.Marker(
            [loc_data['lat'], loc_data['lon']],
            popup=f"{loc}<br>{selected_parameter}: {loc_data[selected_parameter]:.2f}",
            tooltip=loc
        ).add_to(m)
    
    # Display map
    folium_static(m)

with col2:
    st.subheader("Current Statistics")
    current_data = data[data['location'] == selected_location].iloc[-1]
    
    # Display current metrics
    st.metric(
        label="Temperature (°C)",
        value=f"{current_data['temperature']:.1f}",
        delta=f"{current_data['temperature'] - data[data['location'] == selected_location].iloc[-2]['temperature']:.1f}"
    )
    
    st.metric(
        label="Air Quality Index",
        value=f"{current_data['air_quality']:.1f}",
        delta=f"{current_data['air_quality'] - data[data['location'] == selected_location].iloc[-2]['air_quality']:.1f}"
    )
    
    st.metric(
        label="Rainfall (mm)",
        value=f"{current_data['rainfall']:.1f}",
        delta=f"{current_data['rainfall'] - data[data['location'] == selected_location].iloc[-2]['rainfall']:.1f}"
    )

# Trend Analysis
st.subheader("Trend Analysis")
filtered_data = data[
    (data['location'] == selected_location) &
    (data['date'] >= pd.Timestamp(date_range[0])) &
    (data['date'] <= pd.Timestamp(date_range[1]))
]

fig = px.line(
    filtered_data,
    x='date',
    y=selected_parameter,
    title=f"{selected_parameter.title()} Trend for {selected_location}"
)
st.plotly_chart(fig, use_container_width=True)

# Forecast
st.subheader("Forecast")
if st.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        location_data = data[data['location'] == selected_location]
        forecast = create_forecast(location_data, selected_parameter)
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=location_data['date'],
            y=location_data[selected_parameter],
            name='Actual',
            mode='lines'
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))
        
        fig.update_layout(
            title=f"{selected_parameter.title()} Forecast for {selected_location}",
            xaxis_title="Date",
            yaxis_title=selected_parameter.title()
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer with additional information
st.markdown("---")
st.markdown("""
    **About this dashboard:**
    - Includes all state capitals and major union territory capitals of India
    - Data is updated daily
    - Forecasts use Facebook Prophet model
    - Air Quality Index ranges from 0 (Good) to 500 (Hazardous)
    - Temperature and rainfall patterns are adjusted for:
        - Regional variations (North/South/East/West)
        - Seasonal changes (Summer/Winter/Monsoon)
        - Latitude-based temperature differences
""")