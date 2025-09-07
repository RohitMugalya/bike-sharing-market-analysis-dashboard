import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Bike Sharing Market Research",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #2ca02c;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">Bike Sharing Market Research</div>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides insights into bike sharing patterns based on historical data from 2011-2012.
**Note:** Temperature, feel temperature, humidity, and windspeed are normalized between 0 to 1.
""")

# Helper function for polynomial regression
def polynomial_regression_predict(x, y, degree, num_points=200):
    coeffs = np.polyfit(x, y, deg=degree)
    poly_fn = np.poly1d(coeffs)
    
    x_range = np.linspace(np.min(x), np.max(x), num_points)
    y_pred = poly_fn(x_range)
    
    return x_range, y_pred

# Load data
@st.cache_data
def load_data():
    filepath = "data/Bike_sharing_market_research.xlsx"
    data = pd.read_excel(filepath)
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# Define column names for consistency
date = 'date'
season = 'season'
year = 'year'
month = 'month'
holiday = 'holiday'
weekday = 'weekday'
workingday = 'workingday'
weather_situation = 'weather situation'
temperature = 'temperature'
feel_temperature = 'feel temperature'
humidity = 'humidity'
windspeed = 'windspeed'
casual = 'casual'
registered = 'registered'
total = 'total'

# Display basic information
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Date Range", f"{data[date].min().date()} to {data[date].max().date()}")
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Maximum Daily Rentals", f"{data[total].max():,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Minimum Daily Rentals", f"{data[total].min():,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    rentals_2011 = data[data[year] == 2011][total].sum()
    rentals_2012 = data[data[year] == 2012][total].sum()
    pct_change = ((rentals_2012 - rentals_2011) / rentals_2011) * 100
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Yearly Growth (2011-2012)", f"{pct_change:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Time Series Analysis
st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Total Rentals", "User Types", "Moving Average"])

with tab1:
    fig = px.line(
        data,
        x=date,
        y=total,
        title="Total Bike Rentals Over Time",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    color_map = {
        casual: '#1f77b4',
        registered: '#ff7f0e'
    }

    fig = px.line(
        data,
        x=date,
        y=[casual, registered],
        title="Total Bike Rentals by User Type Over Time",
        height=450,
        color_discrete_map=color_map
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">Most rentals are from registered users.</div>', unsafe_allow_html=True)

with tab3:
    moving_avg_total = data[total].rolling(window=10).mean()
    moving_avg_total.name = 'moving_avg_total'

    fig = px.line(
        data,
        x=date,
        y=[total, moving_avg_total],
        title="Total Bike Rentals Over Time (Moving Average)",
        height=450,
        color_discrete_map={total: '#1f77b4', 'moving_avg_total': '#2ca02c'}
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Rentals",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">The seasonality pattern holds for both 2011 and 2012. The number of rentals are low during the winter months December to early March.</div>', unsafe_allow_html=True)

# Area chart
fig = px.area(
    data,
    x=date,
    y=[moving_avg_total],
    title="Total Bike Rentals Over Time (Area Chart)",
    height=400,
    color_discrete_sequence=["#d62728"],
)
fig.update_traces(opacity=0.6)
st.plotly_chart(fig, use_container_width=True)

# Weekly Patterns
st.markdown('<div class="section-header">Weekly Patterns</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_rental_per_weekday = data[[weekday, total]].groupby(weekday).sum()
    day_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    total_rental_per_weekday = total_rental_per_weekday.reindex(day_order)

    fig = px.bar(
        total_rental_per_weekday,
        x=total_rental_per_weekday.index,
        y=total,
        text=total_rental_per_weekday[total],
        title="Total Bike Rentals by Weekday",
        color_discrete_sequence=['#9467bd'],
    )
    fig.update_traces(texttemplate='%{text:,}', textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">The rentals increase from weekday to weekend, with the highest rentals on Thursday to Saturday.</div>', unsafe_allow_html=True)

with col2:
    weather_situation_total = data.groupby(weather_situation)[total].sum()
    fig = px.pie(
        weather_situation_total,
        values=weather_situation_total.values,
        names=weather_situation_total.index,
        title="Proportion of Total Bike Rentals by Weather Situation",
        color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728'],
    )
    st.plotly_chart(fig, use_container_width=True)

# Weather Analysis
st.markdown('<div class="section-header">Weather Impact Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_rentals_weekday_weather = pd.pivot_table(
        data,
        values=total,
        index=weekday,
        columns=weather_situation,
        aggfunc='sum'
    ).loc[day_order]
    
    fig = px.bar(
        total_rentals_weekday_weather,
        y=total_rentals_weekday_weather.columns,
        barmode='group',
        title="Total Rentals by Weekday and Weather",
        labels={"value": "Total Rentals", "variable": "Weather Situation"},
        color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728'],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">The weekday to weekend pattern depends on the weather situation.</div>', unsafe_allow_html=True)

with col2:
    season_weather_rentals = data.pivot_table(
        values=total,
        index=season,
        columns=weather_situation,
        aggfunc='sum'
    )

    fig = px.imshow(
        season_weather_rentals,
        text_auto=True,
        color_continuous_scale='Blues',
        height=500,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis_colorbar=dict(title="Total Rentals"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">Non-winter months and good weather situation have the highest rentals.</div>', unsafe_allow_html=True)

# Environmental Factors
st.markdown('<div class="section-header">Environmental Factors</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    temperature_total_trend_line = polynomial_regression_predict(data[temperature], data[total], degree=2)
    x, y = temperature_total_trend_line

    fig = px.scatter(
        data,
        x=temperature,
        y=total,
        title="Temperature vs Total Rentals",
        labels={"temperature": "Temperature (normalized)", "total": "Total Rentals"},
        color_discrete_sequence=['#1f77b4'],
        height=500,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='#d62728', width=4),
            name='Trend Line (Poly Deg 2)',
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">In general, the higher the temperature, the more the rentals.</div>', unsafe_allow_html=True)

with col2:
    humidity_total_trend_line = polynomial_regression_predict(data[humidity], data[total], degree=2)

    fig = px.scatter(
        data,
        x=humidity,
        y=total,
        color=weather_situation,
        title="Humidity vs Total Rentals",
        labels={"humidity": "Humidity (normalized)", "total": "Total Rentals"},
        color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728'],
        height=500,
    )

    fig.add_trace(
        go.Scatter(
            x=humidity_total_trend_line[0],
            y=humidity_total_trend_line[1],
            mode='lines',
            line=dict(color='#1f77b4', width=4),
            name='Trend Line (Poly Deg 2)',
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box">High humidity corresponds to worst weather situation so low rentals. Humidity between 0.4 to 0.8 has the better rentals.</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
""")