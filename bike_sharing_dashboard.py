import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bike Sharing Market Research",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ff7f0e;
    }
    .insight {
        background-color: var(--background-color, #f8f9fa);
        border: 2px solid #4a90e2;
        color: var(--text-color, #2c3e50);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.15);
        position: relative;
        transition: all 0.3s ease;
    }
    .insight::before {
        content: "💡";
        font-size: 1.3em;
        margin-right: 0.8rem;
        color: #4a90e2;
    }
    .insight:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.25);
    }
    @media (prefers-color-scheme: dark) {
        .insight {
            --background-color: #2d3748;
            --text-color: #e2e8f0;
            border-color: #63b3ed;
        }
        .insight::before {
            color: #63b3ed;
        }
    }
    .data-explorer-btn {
        background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .data-explorer-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4);
    }
    .data-explorer-btn.active {
        background: linear-gradient(135deg, #ff7f0e 0%, #d62728 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Bike Sharing Market Research Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides insights into bike sharing patterns based on historical data from 2011-2012.
Explore trends, seasonal patterns, and weather impacts on bike rentals.
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

# Sidebar filters
st.sidebar.header('Filters')
year_filter = st.sidebar.multiselect(
    'Select Year',
    options=sorted(data[year].unique()),
    default=sorted(data[year].unique())
)

season_filter = st.sidebar.multiselect(
    'Select Season',
    options=sorted(data[season].unique()),
    default=sorted(data[season].unique())
)

weather_filter = st.sidebar.multiselect(
    'Select Weather Situation',
    options=sorted(data[weather_situation].unique()),
    default=sorted(data[weather_situation].unique())
)

# Apply filters
filtered_data = data.copy()
if year_filter:
    filtered_data = filtered_data[filtered_data[year].isin(year_filter)]
if season_filter:
    filtered_data = filtered_data[filtered_data[season].isin(season_filter)]
if weather_filter:
    filtered_data = filtered_data[filtered_data[weather_situation].isin(weather_filter)]

st.info("ℹ️ Tip: Close the sidebar (top-left arrow) for a better chart view.")

# Key metrics
st.markdown("## Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rentals", f"{filtered_data[total].sum():,}")

with col2:
    st.metric("Average Daily Rentals", f"{filtered_data[total].mean():.0f}")

with col3:
    max_date = filtered_data.loc[filtered_data[total].idxmax(), date].date()
    st.metric("Peak Rental Day", f"{max_date}")

with col4:
    pct_change = ((filtered_data[filtered_data[year] == 2012][total].sum() - 
                  filtered_data[filtered_data[year] == 2011][total].sum()) / 
                  filtered_data[filtered_data[year] == 2011][total].sum()) * 100
    st.metric("Yearly Growth", f"{pct_change:.1f}%")

# Time Series Analysis
st.markdown('<h2 class="section-header">Time Series Analysis</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Moving Average", "User Segments", "Area Chart", "Total Rentals"])

with tab1:
    moving_avg_total = 'moving_avg_total'
    filtered_data[moving_avg_total] = filtered_data[total].rolling(window=10).mean()

    fig = px.line(
        filtered_data,
        x=date,
        y=[total, moving_avg_total],
        title="Total Bike Rentals Over Time (Moving Average)",
        height=450,
        color_discrete_map={total: '#1f77b4', moving_avg_total: '#ff7f0e'}
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Rentals",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">The seasonality pattern holds for both 2011 and 2012. The number of rentals are low during the winter months December to early March.</div>', unsafe_allow_html=True)

with tab2:
    color_map = {
        casual: '#1f77b4',
        registered: '#ff7f0e'
    }

    fig = px.line(
        filtered_data,
        x=date,
        y=[casual, registered],
        title="Total Bike Rentals by User Type Over Time",
        height=450,
        color_discrete_map=color_map
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">This shows that most rentals are from registered users.</div>', unsafe_allow_html=True)

with tab3:
    fig = px.area(
        filtered_data,
        x=date,
        y=[casual, registered],
        title="Stacked Area Chart - User Types Over Time",
        height=450,
        color_discrete_map=color_map
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="insight">Percentage change in total rentals from 2011 to 2012: {pct_change:.2f}%</div>', unsafe_allow_html=True)

with tab4:
    fig = px.line(
        filtered_data,
        x=date,
        y=total,
        title="Total Bike Rentals Over Time",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# Weekly Patterns
st.markdown('<h2 class="section-header">Weekly Patterns</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_rental_per_weekday = filtered_data[[weekday, total]].groupby(weekday).sum()
    day_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    total_rental_per_weekday = total_rental_per_weekday.reindex(day_order)

    fig = px.bar(
        total_rental_per_weekday,
        x=total_rental_per_weekday.index,
        y=total,
        text=total_rental_per_weekday[total],
        title="Total Bike Rentals by Weekday",
        color_discrete_sequence=['#1f77b4'],
    )
    fig.update_layout(yaxis={'range': (440_000, 495_000)})
    fig.update_traces(texttemplate='%{text:,}', textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">The rentals increase from weekday to weekend, with the highest rentals on Thursday to Saturday.</div>', unsafe_allow_html=True)

with col2:
    weather_situation_total = filtered_data.groupby(weather_situation)[total].sum()

    fig = px.pie(
        weather_situation_total,
        values=weather_situation_total.values,
        names=weather_situation_total.index,
        title="Proportion of Total Bike Rentals by Weather Situation",
        color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728'],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">People are very less likely to take rental during worst weather situation.</div>', unsafe_allow_html=True)

# Weather and Seasonal Analysis
st.markdown('<h2 class="section-header">Weather and Seasonal Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_rentals_weekday_weather = pd.pivot_table(
        filtered_data,
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
        color_discrete_sequence=['green', 'orange', 'red']
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">The weekday to weekend pattern depends on the weather situation.</div>', unsafe_allow_html=True)

with col2:
    season_weather_rentals = filtered_data.pivot_table(
        values=total,
        index=season,
        columns=weather_situation,
        aggfunc='sum'
    )

    fig = px.imshow(
        season_weather_rentals,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Rentals by Season and Weather Situation"
    )
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis_colorbar=dict(title="Total Rentals"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">Non-winter months and good weather situation have the highest rentals.</div>', unsafe_allow_html=True)

# Environmental Factors
st.markdown('<h2 class="section-header">Environmental Factors</h2>', unsafe_allow_html=True)

# Environmental factor selection
env_factor = st.selectbox(
    "Select Environmental Factor to Analyze:",
    options=["Temperature", "Humidity"],
    index=0
)

if env_factor == "Temperature":
    # Temperature analysis
    temperature_total_trend_line = polynomial_regression_predict(
        filtered_data[temperature], filtered_data[total], degree=2
    )
    x, y = temperature_total_trend_line

    fig = px.scatter(
        filtered_data,
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
            line=dict(color='#ff7f0e', width=4),
            name='Trend Line (Poly Deg 2)',
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">In-general the higher the temperature, the more the rentals.</div>', unsafe_allow_html=True)

else:
    # Humidity analysis
    humidity_total_trend_line = polynomial_regression_predict(
        filtered_data[humidity], filtered_data[total], degree=2
    )

    fig = px.scatter(
        filtered_data,
        x=humidity,
        y=total,
        color=weather_situation,
        title="Humidity vs Total Rentals",
        labels={"humidity": "Humidity (normalized)", "total": "Total Rentals"},
        color_continuous_scale=["green", "orange", "red"],
        height=500,
    )

    fig.add_trace(
        go.Scatter(
            x=humidity_total_trend_line[0],
            y=humidity_total_trend_line[1],
            mode='lines',
            line=dict(color='darkcyan', width=4),
            name='Trend Line (Poly Deg 2)',
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">High humidity corresponds to worst weather situation so low rentals. Humidity between 0.4 to 0.8 has the better rentals.</div>', unsafe_allow_html=True)

# Data Explorer
st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)

# Initialize session state for button toggle
if 'show_dataset' not in st.session_state:
    st.session_state.show_dataset = False

# Custom button with theme-aligned styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    button_text = "🔍 Hide Dataset" if st.session_state.show_dataset else "📊 Show Complete Dataset"
    button_class = "active" if st.session_state.show_dataset else ""
    
    if st.button(button_text, key="dataset_toggle", use_container_width=True):
        st.session_state.show_dataset = not st.session_state.show_dataset

if st.session_state.show_dataset:
    st.markdown("### Complete Bike Sharing Dataset")
    st.dataframe(filtered_data, use_container_width=True)
    st.markdown("**Note:** Temperature, feel temperature, humidity, and windspeed are normalized between 0 to 1.")
    st.markdown(f"**Dataset Info:** {len(filtered_data)} records displayed based on current filters.")

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
""")