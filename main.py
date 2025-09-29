import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from google import genai

# Configure Streamlit page
st.set_page_config(page_title="FloatChat", layout="wide")

# Initialize session state
if 'intent' not in st.session_state:
    st.session_state.intent = None
if 'question' not in st.session_state:
    st.session_state.question = None
if 'filter_key' not in st.session_state:
    st.session_state.filter_key = 0

# Main header
st.title("FloatChat")
st.subheader("Oceanographic Data Visualization with AI")

# Configure Gemini API
def configure_gemini():
    """Configure Gemini AI with API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY environment variable is missing. Please add your Gemini API key.")
        st.stop()
    
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {str(e)}")
        st.stop()

# Initialize Gemini client
gemini_client = configure_gemini()

# Load oceanographic data
@st.cache_data
def load_data():
    """Load oceanographic data from CSV file"""
    try:
        df = pd.read_csv("data.csv")
        return df
    except FileNotFoundError:
        st.error("❌ data.csv file not found. Please ensure the data file exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# File upload functionality
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload custom oceanographic data (CSV)",
    type=['csv'],
    help="Upload a CSV file with columns: date, latitude, longitude, temperature, salinity"
)

# Load data (either uploaded or default)
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Normalize column headers (case-insensitive, strip whitespace)
        df.columns = df.columns.str.strip().str.lower()
        
        # Validate required columns
        required_columns = ['date', 'latitude', 'longitude', 'temperature', 'salinity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.sidebar.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.sidebar.info("📋 Required columns: date, latitude, longitude, temperature, salinity")
            df = load_data()  # Fall back to default data
        else:
            st.sidebar.success(f"✅ Successfully loaded {len(df)} data points from uploaded file")
            
            # Validate data types and ranges
            try:
                # Parse dates with error tolerance
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_dates = df['date'].isna()
                if invalid_dates.any():
                    st.sidebar.warning(f"⚠️ {invalid_dates.sum()} rows have invalid dates and will be excluded")
                    df = df[~invalid_dates]
                
                # Convert numeric columns
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
                df['salinity'] = pd.to_numeric(df['salinity'], errors='coerce')
                
                # Check for reasonable ranges
                lat_valid = df['latitude'].between(-90, 90)
                lon_valid = df['longitude'].between(-180, 180)
                temp_valid = df['temperature'].between(-5, 50)  # Reasonable ocean temps
                sal_valid = df['salinity'].between(0, 50)  # Reasonable salinity range
                
                invalid_rows = ~(lat_valid & lon_valid & temp_valid & sal_valid)
                if invalid_rows.any():
                    st.sidebar.warning(f"⚠️ {invalid_rows.sum()} rows have invalid data ranges and will be excluded")
                    df = df[~invalid_rows]
                
                if len(df) == 0:
                    st.sidebar.error("❌ No valid data rows found after validation")
                    df = load_data()  # Fall back to default data
                    
            except Exception as e:
                st.sidebar.error(f"❌ Data validation error: {str(e)}")
                df = load_data()  # Fall back to default data
                
    except Exception as e:
        st.sidebar.error(f"❌ Error reading uploaded file: {str(e)}")
        df = load_data()  # Fall back to default data
else:
    df = load_data()

# Convert date column to datetime for filtering (if not already done during upload)
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df['date'].isna()
        if invalid_dates.any():
            st.warning(f"⚠️ {invalid_dates.sum()} rows have invalid dates and will be excluded from analysis")
            df = df[~invalid_dates]
        
        if len(df) == 0:
            st.error("❌ No valid dates found in the dataset")
            st.info("💡 Please ensure dates are in a recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.)")
            st.stop()
    except Exception as e:
        st.error(f"❌ Critical error parsing dates: {str(e)}")
        st.info("💡 Please check your date format and try again")
        st.stop()

# Sidebar for data filtering
st.sidebar.header("🔍 Data Filters")

# Date range filter
st.sidebar.subheader("📅 Date Range")
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key=f"date_range_{st.session_state.filter_key}"
)

# Geographic bounds filter
st.sidebar.subheader("🌍 Geographic Bounds")

# Latitude bounds
lat_min, lat_max = float(df['latitude'].min()), float(df['latitude'].max())
lat_range = st.sidebar.slider(
    "Latitude range:",
    min_value=lat_min,
    max_value=lat_max,
    value=(lat_min, lat_max),
    step=0.1,
    format="%.1f°",
    key=f"lat_range_{st.session_state.filter_key}"
)

# Longitude bounds  
lon_min, lon_max = float(df['longitude'].min()), float(df['longitude'].max())
lon_range = st.sidebar.slider(
    "Longitude range:",
    min_value=lon_min,
    max_value=lon_max,
    value=(lon_min, lon_max),
    step=0.1,
    format="%.1f°",
    key=f"lon_range_{st.session_state.filter_key}"
)

# Apply filters
filtered_df = df.copy()

# Filter by date range
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= start_date) & 
        (filtered_df['date'].dt.date <= end_date)
    ]
elif len(date_range) == 1:
    # Single date selected - filter to exact date
    selected_date = date_range[0]
    filtered_df = filtered_df[filtered_df['date'].dt.date == selected_date]

# Filter by geographic bounds
filtered_df = filtered_df[
    (filtered_df['latitude'] >= lat_range[0]) & 
    (filtered_df['latitude'] <= lat_range[1]) &
    (filtered_df['longitude'] >= lon_range[0]) & 
    (filtered_df['longitude'] <= lon_range[1])
]

# Display filtered data info
st.write(f"📊 Showing {len(filtered_df)} of {len(df)} oceanographic data points (filtered)")

# Show filter summary
if len(filtered_df) != len(df):
    st.info(f"🔽 Filters applied: Date range {date_range[0] if len(date_range) == 2 else 'all'} to {date_range[1] if len(date_range) == 2 else 'all'}, Lat {lat_range[0]:.1f}° to {lat_range[1]:.1f}°, Lon {lon_range[0]:.1f}° to {lon_range[1]:.1f}°")

# Add reset filters button
if len(filtered_df) != len(df):
    if st.sidebar.button("🔄 Reset All Filters"):
        # Reset filters by incrementing the key, which creates new widgets with default values
        st.session_state.filter_key += 1
        st.rerun()

# Gemini intent function
def get_intent(user_question):
    """
    Analyze user question using Gemini AI to determine intent.
    Returns specific intent for complex oceanographic analysis
    """
    try:
        prompt = f"""
        Analyze the following user question about oceanographic data and respond with ONLY one word from this list:
        'temperature', 'salinity', 'correlation', 'trend', 'comparison', 'statistics', 'unknown'

        Classification Rules:
        - temperature: Questions about temperature, heat, thermal, warming, cooling
        - salinity: Questions about salinity, salt content, saltiness
        - correlation: Questions asking about relationships between variables, correlations, connections
        - trend: Questions about changes over time, patterns, temporal analysis, trends
        - comparison: Questions comparing different locations, regions, or time periods
        - statistics: Questions asking for summary stats, averages, ranges, distributions
        - unknown: Questions that don't fit above categories or are unclear

        Examples:
        "Show temperature data" → temperature
        "What's the salinity like?" → salinity
        "How do temperature and salinity relate?" → correlation
        "Are temperatures increasing over time?" → trend
        "Compare northern vs southern regions" → comparison
        "What are the average values?" → statistics

        User question: {user_question}

        Response (one word only):
        """
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if response.text:
            intent = response.text.strip().lower()
            # Ensure we only return valid intents
            valid_intents = ['temperature', 'salinity', 'correlation', 'trend', 'comparison', 'statistics', 'unknown']
            if intent in valid_intents:
                return intent
            else:
                return 'unknown'
        else:
            return 'unknown'
            
    except Exception as e:
        st.error(f"❌ Error analyzing question with Gemini: {str(e)}")
        return 'unknown'

# User input section
st.markdown("---")
st.subheader("Ask about the oceanographic data")

# Create columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input(
        "Enter your question about the data:",
        placeholder="e.g., 'Show me the temperature distribution' or 'What about salinity levels?'"
    )

with col2:
    st.write("")  # Add some space
    submit_button = st.button("🔍 Submit", type="primary")

# Handle button click
if submit_button and user_question:
    with st.spinner("🤔 Analyzing your question with AI..."):
        intent = get_intent(user_question)
        st.session_state.intent = intent
        st.session_state.question = user_question
    
    st.write(f"**Question:** {user_question}")
    st.write(f"**AI Intent:** {intent}")

elif submit_button and not user_question:
    st.warning("⚠️ Please enter a question before submitting.")

# Display stored question and intent if available
if st.session_state.question and st.session_state.intent:
    if not (submit_button and user_question):  # Only show if not just submitted
        st.write(f"**Last Question:** {st.session_state.question}")
        st.write(f"**AI Intent:** {st.session_state.intent}")

# Create visualizations based on stored intent (real-time updates with filters)
if st.session_state.intent and len(filtered_df) > 0:
    if st.session_state.intent == 'temperature':
        st.subheader("🌡️ Temperature Distribution Map")
        
        # Create temperature scatter mapbox
        fig = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="temperature",
            size="temperature",
            hover_name="date",
            hover_data={
                "temperature": ":.1f°C",
                "salinity": ":.1f",
                "latitude": ":.2f",
                "longitude": ":.2f"
            },
            color_continuous_scale="thermal",
            size_max=20,
            zoom=6,
            title="Ocean Temperature Distribution"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show temperature statistics
        st.subheader("📈 Temperature Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Temperature", f"{filtered_df['temperature'].min():.1f}°C")
        with col2:
            st.metric("Max Temperature", f"{filtered_df['temperature'].max():.1f}°C")
        with col3:
            st.metric("Mean Temperature", f"{filtered_df['temperature'].mean():.1f}°C")
        with col4:
            st.metric("Std Deviation", f"{filtered_df['temperature'].std():.1f}°C")
    
    elif st.session_state.intent == 'salinity':
        st.subheader("🧂 Salinity Distribution Map")
        
        # Create salinity scatter mapbox
        fig = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="salinity",
            size="salinity",
            hover_name="date",
            hover_data={
                "salinity": ":.1f",
                "temperature": ":.1f°C",
                "latitude": ":.2f",
                "longitude": ":.2f"
            },
            color_continuous_scale="blues",
            size_max=20,
            zoom=6,
            title="Ocean Salinity Distribution"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show salinity statistics
        st.subheader("📈 Salinity Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Salinity", f"{filtered_df['salinity'].min():.1f}")
        with col2:
            st.metric("Max Salinity", f"{filtered_df['salinity'].max():.1f}")
        with col3:
            st.metric("Mean Salinity", f"{filtered_df['salinity'].mean():.1f}")
        with col4:
            st.metric("Std Deviation", f"{filtered_df['salinity'].std():.1f}")
    
    elif st.session_state.intent == 'correlation':
        st.subheader("🔗 Temperature-Salinity Correlation Analysis")
        
        # Create correlation scatter plot
        fig = px.scatter(
            filtered_df,
            x="temperature",
            y="salinity",
            color="date",
            hover_data=['latitude', 'longitude'],
            title="Temperature vs Salinity Correlation",
            labels={'temperature': 'Temperature (°C)', 'salinity': 'Salinity'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation coefficient
        correlation = filtered_df['temperature'].corr(filtered_df['salinity'])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        if abs(correlation) > 0.7:
            st.info(f"🔍 Strong {'positive' if correlation > 0 else 'negative'} correlation detected!")
        elif abs(correlation) > 0.3:
            st.info(f"🔍 Moderate {'positive' if correlation > 0 else 'negative'} correlation detected.")
        else:
            st.info("🔍 Weak correlation between temperature and salinity.")
    
    elif st.session_state.intent == 'trend':
        st.subheader("📈 Temporal Trends Analysis")
        
        # Sort by date for trend analysis
        trend_df = filtered_df.sort_values('date')
        
        # Create dual-axis time series plot
        fig = px.line(
            trend_df,
            x="date",
            y="temperature",
            title="Temperature and Salinity Trends Over Time",
            labels={'temperature': 'Temperature (°C)', 'date': 'Date'}
        )
        
        # Add salinity on secondary y-axis
        fig.add_scatter(
            x=trend_df['date'],
            y=trend_df['salinity'],
            mode='lines+markers',
            name='Salinity',
            yaxis='y2'
        )
        
        fig.update_layout(
            yaxis2=dict(title='Salinity', overlaying='y', side='right'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate trend statistics using actual time
        if len(trend_df) > 1:
            # Convert dates to ordinal days for proper time-based calculation
            date_ordinal = pd.to_numeric(trend_df['date'])
            time_span_days = (trend_df['date'].max() - trend_df['date'].min()).days
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if time_span_days > 0:
                    temp_slope = np.polyfit(date_ordinal, trend_df['temperature'], 1)[0]
                    # Convert to per-day units 
                    temp_slope_per_day = temp_slope * 86400000000000  # nanoseconds to days conversion
                    st.metric("Temperature Trend", f"{temp_slope_per_day:.6f}°C/day")
                else:
                    st.metric("Temperature Trend", "N/A (single day)")
            
            with col2:
                if time_span_days > 0:
                    sal_slope = np.polyfit(date_ordinal, trend_df['salinity'], 1)[0]
                    sal_slope_per_day = sal_slope * 86400000000000  # nanoseconds to days conversion
                    st.metric("Salinity Trend", f"{sal_slope_per_day:.6f}/day")
                else:
                    st.metric("Salinity Trend", "N/A (single day)")
                    
            with col3:
                st.metric("Time Span", f"{time_span_days} days")
        else:
            st.info("Need at least 2 data points for trend analysis")
    
    elif st.session_state.intent == 'comparison':
        st.subheader("🗺️ Regional Comparison Analysis")
        
        # Split data into northern and southern regions based on median latitude
        median_lat = filtered_df['latitude'].median()
        northern = filtered_df[filtered_df['latitude'] >= median_lat]
        southern = filtered_df[filtered_df['latitude'] < median_lat]
        
        # Create comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Northern Region**")
            if len(northern) > 0:
                st.metric("Avg Temperature", f"{northern['temperature'].mean():.1f}°C")
                st.metric("Avg Salinity", f"{northern['salinity'].mean():.1f}")
                st.metric("Data Points", f"{len(northern)}")
            else:
                st.write("No data in northern region")
        
        with col2:
            st.write("**Southern Region**")
            if len(southern) > 0:
                st.metric("Avg Temperature", f"{southern['temperature'].mean():.1f}°C")
                st.metric("Avg Salinity", f"{southern['salinity'].mean():.1f}")
                st.metric("Data Points", f"{len(southern)}")
            else:
                st.write("No data in southern region")
        
        # Box plots for comparison
        if len(northern) > 0 and len(southern) > 0:
            comparison_data = []
            for _, row in northern.iterrows():
                comparison_data.append({'Region': 'Northern', 'Temperature': row['temperature'], 'Salinity': row['salinity']})
            for _, row in southern.iterrows():
                comparison_data.append({'Region': 'Southern', 'Temperature': row['temperature'], 'Salinity': row['salinity']})
            
            comp_df = pd.DataFrame(comparison_data)
            
            fig = px.box(comp_df, x='Region', y='Temperature', title='Temperature Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)
    
    elif st.session_state.intent == 'statistics':
        st.subheader("📊 Comprehensive Statistical Analysis")
        
        # Descriptive statistics
        stats_df = filtered_df[['temperature', 'salinity']].describe()
        st.write("**Descriptive Statistics**")
        st.dataframe(stats_df, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_temp = px.histogram(
                filtered_df, 
                x='temperature', 
                nbins=15,
                title='Temperature Distribution',
                labels={'temperature': 'Temperature (°C)'}
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            fig_sal = px.histogram(
                filtered_df, 
                x='salinity', 
                nbins=15,
                title='Salinity Distribution'
            )
            st.plotly_chart(fig_sal, use_container_width=True)
    
    else:  # st.session_state.intent == 'unknown'
        st.warning("⚠️ I couldn't understand your question. Please try asking about oceanographic data analysis.")
        st.info("💡 **Examples of questions you can ask:**")
        st.write("• 'Show me temperature data' - Temperature mapping")
        st.write("• 'What about salinity levels?' - Salinity visualization")
        st.write("• 'How do temperature and salinity relate?' - Correlation analysis")
        st.write("• 'Are temperatures changing over time?' - Trend analysis")
        st.write("• 'Compare different regions' - Regional comparison")
        st.write("• 'Show me statistical summaries' - Statistical analysis")

elif st.session_state.intent and len(filtered_df) == 0:
    st.warning("⚠️ No data points match the current filters. Please adjust your date range or geographic bounds.")
    st.info("💡 Try expanding the date range or geographic area to see more data.")

# Show raw data section
st.markdown("---")
with st.expander("📋 View Raw Data"):
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
        
        # Add download button for the filtered data
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_data,
            file_name="filtered_oceanographic_data.csv",
            mime="text/csv"
        )
    else:
        st.write("No data to display with current filters.")

# Footer information
st.markdown("---")
st.markdown("*FloatChat - Oceanographic Data Visualization powered by Gemini AI*")
