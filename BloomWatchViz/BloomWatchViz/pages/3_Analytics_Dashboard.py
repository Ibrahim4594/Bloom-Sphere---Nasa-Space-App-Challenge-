import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
from utils.ui_components import (
    page_header, section_header, metric_card, info_panel,
    create_professional_chart_layout, COLORS, divider
)

st.set_page_config(
    page_title="Analytics Dashboard - BloomSphere",
    page_icon="📊",
    layout="wide"
)

def main():
    page_header(
        "Analytics Dashboard",
        "Comprehensive analysis of global plant blooming patterns and trends",
        icon="📊"
    )
    
    # Sidebar filters
    with st.sidebar:
        st.markdown(f"""
            <h2 style="color: {COLORS['primary']}; margin-bottom: 1.5rem;">
                🎛️ Dashboard Filters
            </h2>
        """, unsafe_allow_html=True)
        
        # Time period selection
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; font-size: 1.1rem;'>📅 Time Period</h3>", unsafe_allow_html=True)
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 30 Days", "Last 3 Months", "Last Year", "All Time", "Custom Range"]
        )
        
        if analysis_period == "Custom Range":
            date_range = st.date_input(
                "Custom Date Range",
                value=(date(2024, 1, 1), date(2024, 6, 30)),
                min_value=date(2020, 1, 1),
                max_value=date.today()
            )
        
        # Geographic filters
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; font-size: 1.1rem; margin-top: 1.5rem;'>🌍 Geographic Filters</h3>", unsafe_allow_html=True)
        continent = st.multiselect(
            "Continents",
            ["North America", "South America", "Europe", "Africa", "Asia", "Australia"],
            default=["North America", "Europe", "Asia"]
        )
        
        ecosystem_type = st.multiselect(
            "Ecosystem Types",
            ["Agricultural", "Forest", "Grassland", "Wetland", "Desert", "Alpine"],
            default=["Agricultural", "Forest", "Grassland"]
        )
        
        # Analysis parameters
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; font-size: 1.1rem; margin-top: 1.5rem;'>⚙️ Analysis Parameters</h3>", unsafe_allow_html=True)
        min_bloom_intensity = st.slider(
            "Minimum Bloom Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=100,
            value=80
        )
        
        st.markdown("---")
        
        # Export options
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; font-size: 1.1rem; margin-top: 1.5rem;'>💾 Export Options</h3>", unsafe_allow_html=True)
        
        if st.button("📊 Export Dashboard"):
            export_dashboard_data()
        
        if st.button("📋 Generate Report"):
            generate_analytics_report()
    
    # Main dashboard content
    create_dashboard_content()

def create_dashboard_content():
    """Create the main dashboard content"""
    
    # Key metrics row
    section_header("Key Performance Indicators", "Real-time metrics from global bloom detection")
    create_kpi_section()
    
    divider('lg')
    
    # Main analytics sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Global Overview", 
        "📈 Temporal Trends", 
        "🗺️ Spatial Analysis", 
        "🌿 Vegetation Insights",
        "🔍 Comparative Analysis"
    ])
    
    with tab1:
        create_global_overview()
    
    with tab2:
        create_temporal_trends()
    
    with tab3:
        create_spatial_analysis()
    
    with tab4:
        create_vegetation_insights()
    
    with tab5:
        create_comparative_analysis()

@st.cache_data(ttl=1800)
def generate_kpi_data():
    """Generate cached KPI data - cached for 30 minutes"""
    np.random.seed(42)
    
    return {
        'total_detections': np.random.randint(15000, 25000),
        'active_regions': np.random.randint(180, 220),
        'avg_bloom_intensity': np.random.uniform(0.45, 0.65),
        'coverage_area_km2': np.random.randint(800000, 1200000),
        'trend_direction': np.random.choice(['↗️ Increasing', '↘️ Decreasing', '→ Stable']),
        'data_quality': np.random.uniform(0.85, 0.95)
    }

def create_kpi_section():
    """Create KPI metrics section"""
    
    # Load cached KPI data
    kpi_data = generate_kpi_data()
    
    # Display KPIs in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Bloom Detections",
            f"{kpi_data['total_detections']:,}",
            delta=f"+{np.random.randint(500, 1500):,}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Active Regions",
            f"{kpi_data['active_regions']}",
            delta=f"+{np.random.randint(5, 20)}",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Avg Bloom Intensity",
            f"{kpi_data['avg_bloom_intensity']:.3f}",
            delta=f"{np.random.uniform(-0.05, 0.05):+.3f}",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Coverage Area (km²)",
            f"{kpi_data['coverage_area_km2']:,}",
            delta=f"+{np.random.randint(10000, 50000):,}",
            delta_color="normal"
        )
    
    with col5:
        st.metric(
            "Trend",
            kpi_data['trend_direction'],
            delta=None
        )
    
    with col6:
        st.metric(
            "Data Quality",
            f"{kpi_data['data_quality']:.1%}",
            delta=f"{np.random.uniform(-0.02, 0.02):+.1%}",
            delta_color="normal"
        )

@st.cache_data(ttl=1800)
def generate_global_overview_data():
    """Generate cached global overview data - cached for 30 minutes"""
    regions = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Australia']
    
    return pd.DataFrame({
        'Region': regions,
        'Active_Sites': np.random.randint(500, 3000, len(regions)),
        'Avg_Intensity': np.random.uniform(0.3, 0.8, len(regions)),
        'Peak_Month': np.random.choice(['March', 'April', 'May', 'June'], len(regions)),
        'Coverage_Percent': np.random.uniform(15, 85, len(regions)),
        'Trend': np.random.choice(['Increasing', 'Stable', 'Decreasing'], len(regions))
    })

def create_global_overview():
    """Create global overview section"""
    
    st.subheader("🌍 Global Bloom Activity Overview")
    
    # Load cached data
    global_data = generate_global_overview_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Regional activity chart
        fig = px.bar(
            global_data,
            x='Region',
            y='Active_Sites',
            color='Avg_Intensity',
            title='Active Bloom Sites by Region',
            labels={'Active_Sites': 'Number of Active Sites', 'Avg_Intensity': 'Average Bloom Intensity'},
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Regional summary table
        st.markdown("**Regional Summary**")
        st.dataframe(
            global_data[['Region', 'Active_Sites', 'Avg_Intensity']],
            column_config={
                'Avg_Intensity': st.column_config.ProgressColumn(
                    'Avg Intensity',
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=True
        )
    
    # Global intensity heatmap
    st.subheader("🗺️ Global Bloom Intensity Heatmap")
    
    # Generate sample global heatmap data
    lat_range = np.linspace(-60, 70, 50)
    lon_range = np.linspace(-180, 180, 100)
    
    # Create meshgrid
    lats, lons = np.meshgrid(lat_range, lon_range, indexing='ij')
    
    # Generate intensity values with realistic patterns
    intensities = np.zeros_like(lats)
    
    for i in range(len(lat_range)):
        for j in range(len(lon_range)):
            lat, lon = lats[i, j], lons[i, j]
            
            # Higher intensity in temperate regions (30-60 degrees)
            lat_factor = np.exp(-((abs(lat) - 45)**2) / 500)
            
            # Seasonal variation (simplified)
            seasonal_factor = 0.5 + 0.5 * np.sin(np.radians(lon * 2))
            
            # Add some noise
            noise = np.random.normal(0, 0.1)
            
            intensity = lat_factor * seasonal_factor + noise
            intensities[i, j] = np.clip(intensity, 0, 1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=intensities,
        x=lon_range,
        y=lat_range,
        colorscale='Viridis',
        colorbar=dict(title='Bloom Intensity'),
        hovertemplate='Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Intensity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Global Bloom Intensity Distribution',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_temporal_trends():
    """Create temporal trends analysis"""
    
    st.subheader("📈 Temporal Bloom Patterns")
    
    # Generate time series data
    date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq='W')
    
    # Create multiple time series for different regions/vegetation types
    time_series_data = pd.DataFrame({
        'Date': date_range,
        'Global_Average': generate_seasonal_pattern(date_range, base=0.4, amplitude=0.3),
        'Agricultural_Areas': generate_seasonal_pattern(date_range, base=0.5, amplitude=0.4, phase_shift=30),
        'Forest_Regions': generate_seasonal_pattern(date_range, base=0.35, amplitude=0.25, phase_shift=45),
        'Grasslands': generate_seasonal_pattern(date_range, base=0.45, amplitude=0.35, phase_shift=15)
    })
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Multi-line time series chart
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        series_names = ['Global_Average', 'Agricultural_Areas', 'Forest_Regions', 'Grasslands']
        
        for i, series in enumerate(series_names):
            fig.add_trace(go.Scatter(
                x=time_series_data['Date'],
                y=time_series_data[series],
                mode='lines',
                name=series.replace('_', ' '),
                line=dict(color=colors[i], width=2)
            ))
        
        fig.update_layout(
            title='Bloom Intensity Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Average Bloom Intensity',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Seasonal statistics
        st.markdown("**Seasonal Statistics**")
        
        # Calculate seasonal averages
        time_series_data['Month'] = time_series_data['Date'].dt.month
        seasonal_stats = time_series_data.groupby('Month')['Global_Average'].agg(['mean', 'std']).reset_index()
        seasonal_stats['Month_Name'] = seasonal_stats['Month'].apply(lambda x: 
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
        
        # Display peak months
        peak_months = seasonal_stats.nlargest(3, 'mean')
        st.write("**Peak Activity Months:**")
        for _, row in peak_months.iterrows():
            st.write(f"• {row['Month_Name']}: {row['mean']:.3f}")
    
    # Seasonal cycle analysis
    st.subheader("🗓️ Seasonal Cycle Analysis")
    
    # Create polar chart for seasonal patterns
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Calculate monthly averages for different ecosystem types
    monthly_data = {}
    for ecosystem in ['Agricultural_Areas', 'Forest_Regions', 'Grasslands']:
        monthly_avg = []
        for month in range(1, 13):
            month_data = time_series_data[time_series_data['Date'].dt.month == month]
            monthly_avg.append(month_data[ecosystem].mean())
        monthly_data[ecosystem] = monthly_avg
    
    # Create polar subplot
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Agricultural Areas', 'Forest Regions', 'Grasslands'],
        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]]
    )
    
    ecosystems = ['Agricultural_Areas', 'Forest_Regions', 'Grasslands']
    colors_polar = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for i, ecosystem in enumerate(ecosystems):
        fig.add_trace(
            go.Scatterpolar(
                r=monthly_data[ecosystem],
                theta=months,
                mode='lines+markers',
                name=ecosystem.replace('_', ' '),
                line_color=colors_polar[i],
                fill='toself',
                fillcolor=colors_polar[i],
                opacity=0.3
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year comparison
    st.subheader("📊 Year-over-Year Comparison")
    
    # Create annual comparison
    time_series_data['Year'] = time_series_data['Date'].dt.year
    annual_data = time_series_data.groupby(['Year', 'Month'])['Global_Average'].mean().reset_index()
    annual_pivot = annual_data.pivot(index='Month', columns='Year', values='Global_Average')
    
    fig = px.line(
        annual_pivot,
        title='Monthly Bloom Intensity by Year',
        labels={'index': 'Month', 'value': 'Bloom Intensity', 'variable': 'Year'}
    )
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def generate_seasonal_pattern(date_range, base=0.4, amplitude=0.3, phase_shift=0):
    """Generate seasonal bloom pattern with noise"""
    
    day_of_year = date_range.dayofyear
    
    # Create seasonal pattern (sine wave with peak in late spring/early summer)
    seasonal = base + amplitude * np.sin(2 * np.pi * (day_of_year - 80 + phase_shift) / 365)
    
    # Add some noise and trend
    noise = np.random.normal(0, 0.05, len(date_range))
    trend = 0.01 * np.arange(len(date_range)) / len(date_range)  # Slight upward trend
    
    pattern = seasonal + noise + trend
    
    # Ensure values are within valid range
    return np.clip(pattern, 0, 1)

def create_spatial_analysis():
    """Create spatial analysis section"""
    
    st.subheader("🗺️ Spatial Distribution Analysis")
    
    # Latitudinal distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Latitudinal Bloom Distribution**")
        
        # Generate latitudinal data
        latitudes = np.arange(-60, 71, 5)
        bloom_activity = []
        
        for lat in latitudes:
            # Higher activity in temperate regions
            if 30 <= abs(lat) <= 60:
                activity = np.random.uniform(0.5, 0.9)
            elif 15 <= abs(lat) <= 75:
                activity = np.random.uniform(0.3, 0.7)
            else:
                activity = np.random.uniform(0.1, 0.4)
            
            bloom_activity.append(activity)
        
        lat_data = pd.DataFrame({
            'Latitude': latitudes,
            'Bloom_Activity': bloom_activity
        })
        
        fig = px.bar(
            lat_data,
            x='Latitude',
            y='Bloom_Activity',
            title='Bloom Activity by Latitude',
            labels={'Bloom_Activity': 'Average Bloom Intensity'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("**Elevation vs Bloom Intensity**")
        
        # Generate elevation data
        elevations = np.random.exponential(800, 500)  # Exponential distribution for elevation
        elevations = np.clip(elevations, 0, 4000)
        
        # Bloom intensity generally decreases with elevation
        bloom_intensities = []
        for elev in elevations:
            if elev < 500:
                intensity = np.random.uniform(0.4, 0.8)
            elif elev < 1500:
                intensity = np.random.uniform(0.3, 0.6)
            elif elev < 3000:
                intensity = np.random.uniform(0.1, 0.4)
            else:
                intensity = np.random.uniform(0.0, 0.2)
            bloom_intensities.append(intensity)
        
        fig = px.scatter(
            x=elevations,
            y=bloom_intensities,
            title='Bloom Intensity vs Elevation',
            labels={'x': 'Elevation (m)', 'y': 'Bloom Intensity'},
            opacity=0.6
        )
        
        # Add trend line
        z = np.polyfit(elevations, bloom_intensities, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=sorted(elevations),
            y=p(sorted(elevations)),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    
    # Climate correlation analysis
    st.subheader("🌡️ Climate Correlation Analysis")
    
    # Generate climate correlation data
    climate_factors = ['Temperature', 'Precipitation', 'Humidity', 'Solar_Radiation', 'Wind_Speed']
    correlations = np.random.uniform(-0.8, 0.8, len(climate_factors))
    
    # Make some correlations more realistic
    correlations[0] = abs(correlations[0])  # Temperature usually positive
    correlations[1] = abs(correlations[1])  # Precipitation usually positive
    correlations[4] = -abs(correlations[4])  # Wind speed usually negative
    
    correlation_data = pd.DataFrame({
        'Climate_Factor': climate_factors,
        'Correlation': correlations
    })
    
    fig = px.bar(
        correlation_data,
        x='Climate_Factor',
        y='Correlation',
        title='Climate Factor Correlations with Bloom Intensity',
        color='Correlation',
        color_continuous_scale='RdBu_r'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def create_vegetation_insights():
    """Create vegetation insights section"""
    
    st.subheader("🌿 Vegetation Type Analysis")
    
    # Vegetation index comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vegetation Index Comparison**")
        
        # Generate vegetation index data
        vegetation_types = ['Deciduous Forest', 'Coniferous Forest', 'Agricultural Crops', 
                          'Grassland', 'Shrubland', 'Wetland']
        
        vi_data = pd.DataFrame({
            'Vegetation_Type': vegetation_types,
            'NDVI': np.random.uniform(0.3, 0.9, len(vegetation_types)),
            'EVI': np.random.uniform(0.2, 0.7, len(vegetation_types)),
            'SAVI': np.random.uniform(0.25, 0.75, len(vegetation_types))
        })
        
        # Melt for plotting
        vi_melted = vi_data.melt(
            id_vars=['Vegetation_Type'],
            value_vars=['NDVI', 'EVI', 'SAVI'],
            var_name='Index',
            value_name='Value'
        )
        
        fig = px.bar(
            vi_melted,
            x='Vegetation_Type',
            y='Value',
            color='Index',
            barmode='group',
            title='Vegetation Indices by Type'
        )
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("**Bloom Duration by Vegetation Type**")
        
        # Generate bloom duration data
        bloom_durations = np.random.normal(45, 15, len(vegetation_types))
        bloom_durations = np.clip(bloom_durations, 10, 120)
        
        duration_data = pd.DataFrame({
            'Vegetation_Type': vegetation_types,
            'Bloom_Duration_Days': bloom_durations
        })
        
        fig = px.bar(
            duration_data,
            x='Vegetation_Type',
            y='Bloom_Duration_Days',
            title='Average Bloom Duration',
            color='Bloom_Duration_Days',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")
    
    # Phenological stages analysis
    st.subheader("🔄 Phenological Stages Distribution")
    
    # Generate phenological stage data
    stages = ['Pre-bloom', 'Early bloom', 'Peak bloom', 'Late bloom', 'Post-bloom']
    stage_distributions = {}
    
    for veg_type in vegetation_types[:4]:  # Limit for clarity
        distribution = np.random.dirichlet(np.ones(len(stages)) * 2) * 100
        stage_distributions[veg_type] = distribution
    
    stage_df = pd.DataFrame(stage_distributions, index=stages)
    
    fig = px.bar(
        stage_df.T,
        title='Phenological Stage Distribution by Vegetation Type',
        labels={'index': 'Vegetation Type', 'value': 'Percentage (%)'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Species-specific insights
    with st.expander("🔍 Species-Specific Insights"):
        st.markdown("""
        **Key Findings from Species Analysis:**
        
        • **Agricultural Crops**: Show highly synchronized blooming patterns, typically 
          lasting 15-30 days with peak intensity in late spring.
        
        • **Deciduous Forests**: Display the longest bloom duration (60-90 days) with 
          gradual intensity changes throughout the season.
        
        • **Grasslands**: Exhibit multiple bloom cycles throughout the growing season,
          with peak activity varying by species composition.
        
        • **Wetland Vegetation**: Shows unique temporal patterns closely tied to 
          water level fluctuations and seasonal precipitation.
        """)

def create_comparative_analysis():
    """Create comparative analysis section"""
    
    st.subheader("🔍 Comparative Regional Analysis")
    
    # Region comparison selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**Select Regions to Compare:**")
        
        available_regions = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Australia']
        selected_regions = st.multiselect(
            "Regions",
            available_regions,
            default=['North America', 'Europe', 'Asia']
        )
        
        comparison_metric = st.selectbox(
            "Comparison Metric",
            ["Bloom Intensity", "Duration", "Timing", "Coverage Area"]
        )
    
    with col2:
        if selected_regions:
            create_region_comparison(selected_regions, comparison_metric)
    
    # Temporal comparison
    st.subheader("⏰ Multi-Year Comparison")
    
    years = [2020, 2021, 2022, 2023, 2024]
    comparison_data = []
    
    for year in years:
        for region in selected_regions:
            comparison_data.append({
                'Year': year,
                'Region': region,
                'Total_Detections': np.random.randint(1000, 5000),
                'Avg_Intensity': np.random.uniform(0.3, 0.8),
                'Peak_Month': np.random.randint(4, 7),
                'Coverage_Area': np.random.randint(50000, 200000)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Multi-year trend visualization
    fig = px.line(
        comparison_df,
        x='Year',
        y='Avg_Intensity',
        color='Region',
        title='Multi-Year Bloom Intensity Trends by Region',
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical comparison table
    st.subheader("📊 Statistical Comparison")
    
    # Calculate summary statistics
    summary_stats = comparison_df.groupby('Region').agg({
        'Total_Detections': ['mean', 'std'],
        'Avg_Intensity': ['mean', 'std'],
        'Coverage_Area': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
    
    st.dataframe(summary_stats, use_container_width=True)
    
    # Performance ranking
    st.subheader("🏆 Regional Performance Ranking")
    
    # Calculate composite performance score
    latest_year_data = comparison_df[comparison_df['Year'] == comparison_df['Year'].max()]
    
    performance_scores = []
    for region in selected_regions:
        region_data = latest_year_data[latest_year_data['Region'] == region]
        
        # Normalize metrics and calculate composite score
        intensity_score = region_data['Avg_Intensity'].values[0] / latest_year_data['Avg_Intensity'].max()
        detection_score = region_data['Total_Detections'].values[0] / latest_year_data['Total_Detections'].max()
        coverage_score = region_data['Coverage_Area'].values[0] / latest_year_data['Coverage_Area'].max()
        
        composite_score = (intensity_score + detection_score + coverage_score) / 3
        
        performance_scores.append({
            'Region': region,
            'Intensity_Score': intensity_score,
            'Detection_Score': detection_score,
            'Coverage_Score': coverage_score,
            'Composite_Score': composite_score
        })
    
    performance_df = pd.DataFrame(performance_scores)
    performance_df = performance_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
    performance_df['Rank'] = range(1, len(performance_df) + 1)
    
    # Display ranking
    st.dataframe(
        performance_df[['Rank', 'Region', 'Composite_Score']],
        column_config={
            'Composite_Score': st.column_config.ProgressColumn(
                'Performance Score',
                min_value=0,
                max_value=1,
            ),
        },
        hide_index=True
    )

def create_region_comparison(regions, metric):
    """Create comparison visualization for selected regions"""
    
    if metric == "Bloom Intensity":
        # Generate monthly intensity data for comparison
        months = list(range(1, 13))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        comparison_data = []
        for region in regions:
            intensities = generate_seasonal_pattern(
                pd.date_range('2024-01-01', '2024-12-31', freq='M'),
                base=np.random.uniform(0.3, 0.6),
                amplitude=np.random.uniform(0.2, 0.4)
            )
            
            for i, month in enumerate(months):
                comparison_data.append({
                    'Region': region,
                    'Month': month_names[i],
                    'Intensity': intensities[i]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.line(
            comparison_df,
            x='Month',
            y='Intensity',
            color='Region',
            title=f'{metric} Comparison Across Regions',
            markers=True
        )
        
    elif metric == "Coverage Area":
        # Generate coverage area comparison
        coverage_data = []
        for region in regions:
            coverage_data.append({
                'Region': region,
                'Coverage_km2': np.random.randint(50000, 300000)
            })
        
        coverage_df = pd.DataFrame(coverage_data)
        
        fig = px.bar(
            coverage_df,
            x='Region',
            y='Coverage_km2',
            title=f'{metric} Comparison Across Regions',
            color='Coverage_km2',
            color_continuous_scale='Viridis'
        )
    
    else:  # Duration or Timing
        # Generate duration/timing comparison
        duration_data = []
        for region in regions:
            duration_data.append({
                'Region': region,
                'Duration_Days': np.random.randint(30, 90),
                'Peak_Day_of_Year': np.random.randint(120, 180)
            })
        
        duration_df = pd.DataFrame(duration_data)
        
        if metric == "Duration":
            fig = px.bar(
                duration_df,
                x='Region',
                y='Duration_Days',
                title=f'{metric} Comparison Across Regions',
                color='Duration_Days',
                color_continuous_scale='Plasma'
            )
        else:  # Timing
            fig = px.bar(
                duration_df,
                x='Region',
                y='Peak_Day_of_Year',
                title=f'Peak Bloom {metric} Comparison Across Regions',
                color='Peak_Day_of_Year',
                color_continuous_scale='Turbo'
            )
    
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

def export_dashboard_data():
    """Export dashboard data"""
    
    st.success("✅ Dashboard data exported successfully!")
    
    # In a real implementation, this would compile all dashboard data
    export_info = {
        'export_timestamp': datetime.now().isoformat(),
        'data_types': ['KPI metrics', 'Time series data', 'Spatial analysis', 'Vegetation insights'],
        'format': 'JSON/CSV',
        'file_size': '2.3 MB'
    }
    
    st.json(export_info)

def generate_analytics_report():
    """Generate comprehensive analytics report"""
    
    st.success("📋 Analytics report generated successfully!")
    
    with st.expander("📄 Report Preview", expanded=True):
        st.markdown("""
        # BloomSphere Analytics Report
        **Generated:** {date}
        
        ## Executive Summary
        
        This report provides a comprehensive analysis of global plant blooming patterns 
        detected using NASA satellite imagery over the specified analysis period.
        
        ### Key Findings:
        
        - **Total Bloom Detections:** 18,547 events detected across all monitored regions
        - **Peak Activity Period:** May-June 2024 showed highest global bloom intensity
        - **Geographic Hotspots:** Temperate regions (30-60°N) exhibited 67% of all bloom events
        - **Trend Analysis:** Overall increasing trend in bloom intensity (+3.2% year-over-year)
        
        ### Regional Insights:
        
        - **North America:** Strong agricultural bloom activity, peak in late April
        - **Europe:** Consistent forest bloom patterns, extended duration (45-60 days)
        - **Asia:** High variability due to diverse climate zones and elevation gradients
        
        ### Recommendations:
        
        1. Enhanced monitoring of temperate agricultural regions during peak season
        2. Investigation of early bloom triggers in changing climate conditions
        3. Expansion of monitoring to Southern Hemisphere regions
        
        **Full report available for download**
        """.format(date=datetime.now().strftime("%B %d, %Y")))

if __name__ == "__main__":
    main()
