"""
Comparative Analysis Page for BloomSphere
Multi-temporal and multi-regional bloom pattern comparisons
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Comparative Analysis - BloomSphere",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Comparative Bloom Analysis")
st.markdown("""
Compare bloom patterns across different time periods and geographic regions.
Analyze temporal trends and spatial variations in vegetation dynamics.
""")

# Tabs for different comparison types
tab1, tab2, tab3 = st.tabs([
    "⏱️ Temporal Comparison",
    "🗺️ Regional Comparison",
    "📈 Trend Analysis"
])

with tab1:
    st.header("Temporal Comparison - Year-over-Year Analysis")
    
    st.markdown("""
    Compare bloom patterns across multiple years to identify trends and anomalies.
    """)
    
    # Region selection
    region_name = st.selectbox(
        "Select Region",
        options=["California Central Valley", "Great Plains", "Amazon Basin", "Sahel Region"]
    )
    
    # Year selection
    col1, col2 = st.columns(2)
    
    with col1:
        years_to_compare = st.multiselect(
            "Select Years to Compare",
            options=[2020, 2021, 2022, 2023, 2024],
            default=[2022, 2023, 2024]
        )
    
    with col2:
        metric_to_compare = st.selectbox(
            "Comparison Metric",
            options=["NDVI", "EVI", "Bloom Coverage %", "Bloom Intensity"]
        )
    
    if years_to_compare:
        # Generate sample temporal data
        @st.cache_data
        def generate_temporal_data(years, region):
            """Generate sample temporal bloom data"""
            data = []
            
            for year in years:
                # Generate seasonal pattern
                days = pd.date_range(
                    start=f'{year}-01-01',
                    end=f'{year}-12-31',
                    freq='W'
                )
                
                for day in days:
                    day_of_year = day.timetuple().tm_yday
                    
                    # Seasonal bloom pattern (peaks around day 120)
                    peak_day = 120 + np.random.randint(-10, 10)
                    distance_from_peak = min(abs(day_of_year - peak_day), 365 - abs(day_of_year - peak_day))
                    
                    # Year-specific variation
                    year_factor = 1.0 + (year - 2022) * 0.05
                    
                    # Calculate value based on metric
                    base_value = 0.35
                    seasonal_bloom = 0.4 * np.exp(-(distance_from_peak ** 2) / (2 * 35 ** 2))
                    noise = np.random.normal(0, 0.03)
                    
                    value = (base_value + seasonal_bloom * year_factor + noise)
                    value = np.clip(value, 0, 1)
                    
                    data.append({
                        'Date': day,
                        'Year': year,
                        'Day_of_Year': day_of_year,
                        'Value': value,
                        'Month': day.month
                    })
            
            return pd.DataFrame(data)
        
        temporal_df = generate_temporal_data(years_to_compare, region_name)
        
        # Plot time series comparison
        st.subheader(f"📈 {metric_to_compare} Time Series Comparison")
        
        fig_temporal = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, year in enumerate(years_to_compare):
            year_data = temporal_df[temporal_df['Year'] == year]
            
            fig_temporal.add_trace(go.Scatter(
                x=year_data['Day_of_Year'],
                y=year_data['Value'],
                mode='lines',
                name=str(year),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig_temporal.update_layout(
            title=f"{metric_to_compare} by Day of Year - {region_name}",
            xaxis_title="Day of Year",
            yaxis_title=metric_to_compare,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Statistical comparison
        st.subheader("📊 Statistical Comparison")
        
        stats_data = []
        for year in years_to_compare:
            year_data = temporal_df[temporal_df['Year'] == year]
            
            stats_data.append({
                'Year': year,
                'Mean': year_data['Value'].mean(),
                'Max': year_data['Value'].max(),
                'Min': year_data['Value'].min(),
                'Std Dev': year_data['Value'].std(),
                'Peak Day': year_data.loc[year_data['Value'].idxmax(), 'Day_of_Year']
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        st.dataframe(
            stats_df,
            use_container_width=True,
            column_config={
                'Year': st.column_config.NumberColumn('Year', format='%d'),
                'Mean': st.column_config.NumberColumn('Mean', format='%.3f'),
                'Max': st.column_config.NumberColumn('Maximum', format='%.3f'),
                'Min': st.column_config.NumberColumn('Minimum', format='%.3f'),
                'Std Dev': st.column_config.NumberColumn('Std Dev', format='%.3f'),
                'Peak Day': st.column_config.NumberColumn('Peak Day', format='%d')
            },
            hide_index=True
        )
        
        # Year-over-year change analysis
        st.subheader("📈 Year-over-Year Changes")
        
        if len(years_to_compare) >= 2:
            col1, col2, col3 = st.columns(3)
            
            # Compare most recent years
            latest_year = max(years_to_compare)
            prev_year = sorted(years_to_compare)[-2]
            
            latest_mean = stats_df[stats_df['Year'] == latest_year]['Mean'].values[0]
            prev_mean = stats_df[stats_df['Year'] == prev_year]['Mean'].values[0]
            
            change = ((latest_mean - prev_mean) / prev_mean) * 100
            
            with col1:
                st.metric(
                    f"{latest_year} Mean {metric_to_compare}",
                    f"{latest_mean:.3f}",
                    delta=f"{change:+.1f}%"
                )
            
            with col2:
                latest_max = stats_df[stats_df['Year'] == latest_year]['Max'].values[0]
                prev_max = stats_df[stats_df['Year'] == prev_year]['Max'].values[0]
                max_change = ((latest_max - prev_max) / prev_max) * 100
                
                st.metric(
                    "Peak Value Change",
                    f"{latest_max:.3f}",
                    delta=f"{max_change:+.1f}%"
                )
            
            with col3:
                latest_peak_day = stats_df[stats_df['Year'] == latest_year]['Peak Day'].values[0]
                prev_peak_day = stats_df[stats_df['Year'] == prev_year]['Peak Day'].values[0]
                day_shift = latest_peak_day - prev_peak_day
                
                st.metric(
                    "Peak Timing Shift",
                    f"Day {int(latest_peak_day)}",
                    delta=f"{day_shift:+d} days"
                )
            
            # Interpretation
            if change > 5:
                st.success(f"✅ Significant increase in vegetation health from {prev_year} to {latest_year}")
            elif change < -5:
                st.warning(f"⚠️ Notable decrease in vegetation health from {prev_year} to {latest_year}")
            else:
                st.info(f"ℹ️ Relatively stable vegetation patterns between {prev_year} and {latest_year}")

with tab2:
    st.header("Regional Comparison - Spatial Analysis")
    
    st.markdown("""
    Compare bloom patterns across different geographic regions.
    """)
    
    # Region selection
    regions_to_compare = st.multiselect(
        "Select Regions to Compare",
        options=[
            "California Central Valley",
            "Great Plains",
            "Amazon Basin",
            "Sahel Region",
            "Australian Outback",
            "Mediterranean Coast"
        ],
        default=["California Central Valley", "Great Plains"]
    )
    
    # Time period
    year_regional = st.selectbox(
        "Select Year",
        options=[2024, 2023, 2022, 2021, 2020],
        index=0
    )
    
    if regions_to_compare:
        # Generate regional comparison data
        @st.cache_data
        def generate_regional_data(regions, year):
            """Generate sample regional bloom data"""
            data = []
            
            # Regional characteristics
            regional_params = {
                "California Central Valley": (0.65, 120, 40),
                "Great Plains": (0.55, 130, 45),
                "Amazon Basin": (0.80, 60, 20),
                "Sahel Region": (0.40, 150, 50),
                "Australian Outback": (0.35, 180, 60),
                "Mediterranean Coast": (0.60, 100, 35)
            }
            
            days = pd.date_range(
                start=f'{year}-01-01',
                end=f'{year}-12-31',
                freq='W'
            )
            
            for region in regions:
                base_ndvi, peak_day, spread = regional_params.get(region, (0.5, 120, 40))
                
                for day in days:
                    day_of_year = day.timetuple().tm_yday
                    distance_from_peak = min(abs(day_of_year - peak_day), 365 - abs(day_of_year - peak_day))
                    
                    seasonal_bloom = (base_ndvi - 0.3) * np.exp(-(distance_from_peak ** 2) / (2 * spread ** 2))
                    noise = np.random.normal(0, 0.02)
                    
                    ndvi = 0.30 + seasonal_bloom + noise
                    ndvi = np.clip(ndvi, 0, 1)
                    
                    data.append({
                        'Date': day,
                        'Region': region,
                        'Day_of_Year': day_of_year,
                        'NDVI': ndvi
                    })
            
            return pd.DataFrame(data)
        
        regional_df = generate_regional_data(regions_to_compare, year_regional)
        
        # Plot regional comparison
        st.subheader(f"🗺️ Regional NDVI Comparison - {year_regional}")
        
        fig_regional = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        
        for i, region in enumerate(regions_to_compare):
            region_data = regional_df[regional_df['Region'] == region]
            
            fig_regional.add_trace(go.Scatter(
                x=region_data['Day_of_Year'],
                y=region_data['NDVI'],
                mode='lines',
                name=region,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig_regional.update_layout(
            title=f"Regional NDVI Patterns - {year_regional}",
            xaxis_title="Day of Year",
            yaxis_title="NDVI",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_regional, use_container_width=True)
        
        # Regional statistics
        st.subheader("📊 Regional Statistics")
        
        regional_stats = []
        for region in regions_to_compare:
            region_data = regional_df[regional_df['Region'] == region]
            
            regional_stats.append({
                'Region': region,
                'Mean NDVI': region_data['NDVI'].mean(),
                'Peak NDVI': region_data['NDVI'].max(),
                'Peak Day': region_data.loc[region_data['NDVI'].idxmax(), 'Day_of_Year'],
                'Duration (days > 0.5)': len(region_data[region_data['NDVI'] > 0.5]) * 7,
                'Variability': region_data['NDVI'].std()
            })
        
        regional_stats_df = pd.DataFrame(regional_stats)
        
        st.dataframe(
            regional_stats_df,
            use_container_width=True,
            column_config={
                'Region': st.column_config.TextColumn('Region', width='large'),
                'Mean NDVI': st.column_config.NumberColumn('Mean NDVI', format='%.3f'),
                'Peak NDVI': st.column_config.NumberColumn('Peak NDVI', format='%.3f'),
                'Peak Day': st.column_config.NumberColumn('Peak Day', format='%d'),
                'Duration (days > 0.5)': st.column_config.NumberColumn('Bloom Duration (days)', format='%d'),
                'Variability': st.column_config.NumberColumn('Variability', format='%.3f')
            },
            hide_index=True
        )
        
        # Regional comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Peak NDVI comparison
            fig_peaks = go.Figure(data=[go.Bar(
                x=regional_stats_df['Region'],
                y=regional_stats_df['Peak NDVI'],
                marker_color='#2ecc71',
                text=regional_stats_df['Peak NDVI'].round(3),
                textposition='outside'
            )])
            
            fig_peaks.update_layout(
                title="Peak NDVI by Region",
                xaxis_title="Region",
                yaxis_title="Peak NDVI",
                height=400
            )
            
            st.plotly_chart(fig_peaks, use_container_width=True)
        
        with col2:
            # Bloom duration comparison
            fig_duration = go.Figure(data=[go.Bar(
                x=regional_stats_df['Region'],
                y=regional_stats_df['Duration (days > 0.5)'],
                marker_color='#3498db',
                text=regional_stats_df['Duration (days > 0.5)'].round(0).astype(int),
                textposition='outside'
            )])
            
            fig_duration.update_layout(
                title="Bloom Duration by Region",
                xaxis_title="Region",
                yaxis_title="Days",
                height=400
            )
            
            st.plotly_chart(fig_duration, use_container_width=True)

with tab3:
    st.header("Long-term Trend Analysis")
    
    st.markdown("""
    Analyze multi-year trends in vegetation indices to identify long-term patterns and climate impacts.
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        trend_region = st.selectbox(
            "Select Region for Trend Analysis",
            options=["California Central Valley", "Great Plains", "Amazon Basin", "Sahel Region"],
            key="trend_region"
        )
    
    with col2:
        trend_years = st.slider(
            "Number of Years",
            min_value=5,
            max_value=20,
            value=10
        )
    
    # Generate long-term trend data
    @st.cache_data
    def generate_trend_data(region, num_years):
        """Generate long-term trend data"""
        data = []
        
        current_year = 2024
        start_year = current_year - num_years
        
        # Long-term trend (slight increase or decrease)
        trend_slope = np.random.choice([-0.01, 0.01, 0.02])
        
        for year in range(start_year, current_year + 1):
            # Annual mean with trend
            base_mean = 0.55 + (year - start_year) * trend_slope
            
            # Annual variability
            annual_mean = base_mean + np.random.normal(0, 0.03)
            annual_max = annual_mean + np.random.uniform(0.15, 0.25)
            annual_min = annual_mean - np.random.uniform(0.15, 0.25)
            
            data.append({
                'Year': year,
                'Mean_NDVI': np.clip(annual_mean, 0, 1),
                'Max_NDVI': np.clip(annual_max, 0, 1),
                'Min_NDVI': np.clip(annual_min, 0, 1)
            })
        
        return pd.DataFrame(data), trend_slope
    
    trend_df, trend_slope = generate_trend_data(trend_region, trend_years)
    
    # Plot long-term trend
    st.subheader(f"📈 {trend_years}-Year Trend - {trend_region}")
    
    fig_trend = go.Figure()
    
    # Mean NDVI trend
    fig_trend.add_trace(go.Scatter(
        x=trend_df['Year'],
        y=trend_df['Mean_NDVI'],
        mode='lines+markers',
        name='Annual Mean NDVI',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=8)
    ))
    
    # Add range
    fig_trend.add_trace(go.Scatter(
        x=trend_df['Year'].tolist() + trend_df['Year'].tolist()[::-1],
        y=trend_df['Max_NDVI'].tolist() + trend_df['Min_NDVI'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='Annual Range'
    ))
    
    # Add trend line
    z = np.polyfit(trend_df['Year'], trend_df['Mean_NDVI'], 1)
    p = np.poly1d(z)
    
    fig_trend.add_trace(go.Scatter(
        x=trend_df['Year'],
        y=p(trend_df['Year']),
        mode='lines',
        name='Trend Line',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig_trend.update_layout(
        title=f"Long-term NDVI Trend",
        xaxis_title="Year",
        yaxis_title="NDVI",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Trend statistics
    st.subheader("📊 Trend Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    overall_change = (trend_df['Mean_NDVI'].iloc[-1] - trend_df['Mean_NDVI'].iloc[0]) / trend_df['Mean_NDVI'].iloc[0] * 100
    
    with col1:
        st.metric(
            "Overall Change",
            f"{overall_change:+.1f}%",
            delta=f"{trend_years} years"
        )
    
    with col2:
        yearly_change = trend_slope * 100
        st.metric(
            "Annual Rate",
            f"{yearly_change:+.2f}%/year"
        )
    
    with col3:
        variability = trend_df['Mean_NDVI'].std()
        st.metric(
            "Variability",
            f"{variability:.3f}"
        )
    
    with col4:
        latest_ndvi = trend_df['Mean_NDVI'].iloc[-1]
        st.metric(
            "Current NDVI",
            f"{latest_ndvi:.3f}"
        )
    
    # Interpretation
    if trend_slope > 0.005:
        st.success(f"✅ **Greening Trend**: Vegetation health has been improving over the past {trend_years} years. This may indicate favorable climate conditions or improved land management.")
    elif trend_slope < -0.005:
        st.warning(f"⚠️ **Browning Trend**: Vegetation health has been declining over the past {trend_years} years. This may signal climate stress, drought, or land degradation.")
    else:
        st.info(f"ℹ️ **Stable Trend**: Vegetation patterns have remained relatively stable over the past {trend_years} years.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About Comparisons")
    
    st.markdown("""
    ### Comparison Types
    
    **Temporal Comparison**
    - Year-over-year analysis
    - Identify seasonal shifts
    - Detect anomalies
    - Track recovery from disturbances
    
    **Regional Comparison**
    - Spatial pattern analysis
    - Climate zone differences
    - Ecosystem comparisons
    - Land use impacts
    
    **Trend Analysis**
    - Long-term vegetation changes
    - Climate impact assessment
    - Greening/browning trends
    - Ecosystem health monitoring
    
    ### Applications
    - Agricultural yield forecasting
    - Climate change monitoring
    - Drought assessment
    - Ecosystem management
    - Policy effectiveness evaluation
    """)
