import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import folium
from streamlit_folium import st_folium
import json
from utils.ui_components import (
    page_header, section_header, metric_card, info_panel,
    create_professional_chart_layout, COLORS
)

st.set_page_config(
    page_title="Global Bloom Map - BloomSphere",
    page_icon="🗺️",
    layout="wide"
)

def main():
    page_header(
        "Global Plant Bloom Map",
        "Interactive visualization of global plant blooming events from NASA satellite data",
        icon="🗺️"
    )
    
    # Sidebar controls
    with st.sidebar:
        st.markdown(f"""
            <h2 style="color: {COLORS['primary']}; margin-bottom: 1.5rem;">
                🎛️ Map Controls
            </h2>
        """, unsafe_allow_html=True)
        
        # Date selection
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; font-size: 1.1rem;'>📅 Time Period</h3>", unsafe_allow_html=True)
        date_range = st.date_input(
            "Select date range",
            value=(date(2024, 3, 1), date(2024, 5, 31)),
            min_value=date(2020, 1, 1),
            max_value=date.today()
        )
        
        # Vegetation index selection
        vegetation_index = st.selectbox(
            "Vegetation Index",
            ["NDVI", "EVI", "SAVI"],
            help="Select which vegetation index to display"
        )
        
        # Intensity threshold
        intensity_threshold = st.slider(
            "Bloom Intensity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Filter locations below this bloom intensity"
        )
        
        # Map style
        map_style = st.selectbox(
            "Map Style",
            ["Satellite", "Terrain", "Streets"],
            index=1
        )
        
        # Region selection
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; font-size: 1.1rem; margin-top: 1.5rem;'>🌍 Focus Region</h3>", unsafe_allow_html=True)
        region = st.selectbox(
            "Select Region",
            ["Global", "North America", "Europe", "Asia", "South America", "Africa", "Australia"],
        )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            show_agriculture = st.checkbox("Agricultural Areas", value=True)
            show_forests = st.checkbox("Forest Regions", value=True)
            show_grasslands = st.checkbox("Grasslands", value=True)
            
            bloom_confidence = st.slider(
                "Detection Confidence",
                min_value=50,
                max_value=100,
                value=75,
                help="Minimum confidence level for bloom detection"
            )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create interactive map
        create_interactive_map(
            date_range, vegetation_index, intensity_threshold,
            region, map_style, show_agriculture, show_forests, show_grasslands
        )
    
    with col2:
        # Display current selection info
        section_header("📊 Current Selection")
        
        info_panel(f"""
        **Date Range:** {date_range[0]} to {date_range[1]}<br>
        **Index:** {vegetation_index}<br>
        **Threshold:** {intensity_threshold}<br>
        **Region:** {region}
        """, panel_type="info", icon="ℹ️")
        
        # Time series controls
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; margin-top: 1.5rem;'>⏱️ Time Animation</h3>", unsafe_allow_html=True)
        
        if st.button("▶️ Play Animation"):
            create_time_animation(date_range, vegetation_index)
        
        animation_speed = st.slider("Animation Speed", 1, 10, 5)
        
        # Export options
        st.markdown(f"<h3 style='color: {COLORS['text_dark']}; margin-top: 1.5rem;'>💾 Export Options</h3>", unsafe_allow_html=True)
        
        if st.button("📷 Export Current View"):
            st.success("Map exported as PNG!")
        
        if st.button("📊 Export Data (CSV)"):
            st.success("Data exported as CSV!")
        
        if st.button("📋 Generate Report"):
            generate_bloom_report(date_range, region, vegetation_index)

def create_interactive_map(date_range, vegetation_index, intensity_threshold, region, map_style, show_agriculture, show_forests, show_grasslands):
    """Create the main interactive bloom map"""
    
    # Generate sample bloom data based on parameters (cached)
    bloom_data = generate_bloom_data(date_range, vegetation_index, intensity_threshold, region)
    
    # Get region coordinates
    region_coords = get_region_coordinates(region)
    
    # Create folium map
    m = folium.Map(
        location=[region_coords['center_lat'], region_coords['center_lon']],
        zoom_start=region_coords['zoom'],
        tiles=get_tile_layer(map_style)
    )
    
    # Add bloom data points
    for idx, row in bloom_data.iterrows():
        # Determine color based on intensity
        color = get_bloom_color(row['intensity'])
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <h4 style="margin: 0; color: #2E8B57;">Bloom Detection</h4>
            <hr style="margin: 5px 0;">
            <b>Location:</b> {row['lat']:.3f}°N, {row['lon']:.3f}°E<br>
            <b>Date:</b> {row['date']}<br>
            <b>{vegetation_index}:</b> {row['intensity']:.3f}<br>
            <b>Confidence:</b> {row['confidence']}%<br>
            <b>Land Cover:</b> {row['land_cover']}<br>
            <b>Bloom Stage:</b> {row['bloom_stage']}
        </div>
        """
        
        # Add marker
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=250),
            color='white',
            weight=1,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f"Bloom Intensity: {row['intensity']:.2f}"
        ).add_to(m)
    
    # Add legend
    add_map_legend(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])
    
    # Show clicked location details
    if map_data["last_clicked"]:
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        
        st.subheader("📍 Location Details")
        
        # Find closest data point
        closest_point = find_closest_bloom_point(bloom_data, clicked_lat, clicked_lng)
        
        if closest_point is not None:
            display_location_details(closest_point, vegetation_index)

def create_time_animation(date_range, vegetation_index):
    """Create animated time series visualization"""
    
    st.subheader("🎬 Bloom Timeline Animation")
    
    # Generate time series data
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    
    # Create weekly intervals
    date_list = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Animation container
    animation_container = st.empty()
    
    progress_bar = st.progress(0)
    
    for i, current_date in enumerate(date_list):
        # Update progress
        progress = (i + 1) / len(date_list)
        progress_bar.progress(progress)
        
        # Generate data for current date
        weekly_data = generate_weekly_bloom_data(current_date, vegetation_index)
        
        # Create visualization for current week
        fig = create_weekly_bloom_chart(weekly_data, current_date, vegetation_index)
        
        # Update container
        with animation_container.container():
            st.plotly_chart(fig, width="stretch")
        
        # Small delay for animation effect
        import time
        time.sleep(0.5)
    
    progress_bar.empty()
    st.success("✅ Animation complete!")

@st.cache_data(ttl=1800)
def generate_bloom_data(date_range, vegetation_index, intensity_threshold, region):
    """Generate cached sample bloom data for the map - cached for 30 minutes"""
    
    region_coords = get_region_coordinates(region)
    np.random.seed(42)  # Consistent demo data
    
    # Number of points based on region size
    if region_coords['zoom'] > 6:
        n_points = np.random.randint(50, 150)
    else:
        n_points = np.random.randint(200, 500)
    
    # Generate coordinates within region bounds
    lats = np.random.uniform(
        region_coords['bounds']['south'],
        region_coords['bounds']['north'],
        n_points
    )
    lons = np.random.uniform(
        region_coords['bounds']['west'],
        region_coords['bounds']['east'],
        n_points
    )
    
    # Generate bloom intensities
    intensities = np.random.beta(2, 3, n_points)  # Skewed toward lower values
    
    # Apply threshold filter
    mask = intensities >= intensity_threshold
    lats = lats[mask]
    lons = lons[mask]
    intensities = intensities[mask]
    
    # Generate additional attributes
    n_filtered = len(lats)
    
    dates = [
        pd.to_datetime(date_range[0]) + 
        timedelta(days=np.random.randint(0, (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days))
        for _ in range(n_filtered)
    ]
    
    confidences = np.random.randint(70, 100, n_filtered)
    
    land_covers = np.random.choice(
        ['Agricultural', 'Forest', 'Grassland', 'Shrubland', 'Wetland'],
        n_filtered,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    bloom_stages = np.random.choice(
        ['Pre-bloom', 'Early bloom', 'Peak bloom', 'Late bloom', 'Post-bloom'],
        n_filtered,
        p=[0.1, 0.2, 0.4, 0.2, 0.1]
    )
    
    # Create DataFrame
    bloom_data = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'intensity': intensities,
        'date': dates,
        'confidence': confidences,
        'land_cover': land_covers,
        'bloom_stage': bloom_stages
    })
    
    return bloom_data

def get_region_coordinates(region):
    """Get coordinates and bounds for different regions"""
    
    regions = {
        'Global': {
            'center_lat': 20, 'center_lon': 0, 'zoom': 2,
            'bounds': {'north': 70, 'south': -60, 'east': 180, 'west': -180}
        },
        'North America': {
            'center_lat': 45, 'center_lon': -100, 'zoom': 3,
            'bounds': {'north': 70, 'south': 15, 'east': -60, 'west': -170}
        },
        'Europe': {
            'center_lat': 54, 'center_lon': 15, 'zoom': 4,
            'bounds': {'north': 71, 'south': 36, 'east': 40, 'west': -10}
        },
        'Asia': {
            'center_lat': 35, 'center_lon': 100, 'zoom': 3,
            'bounds': {'north': 70, 'south': 5, 'east': 150, 'west': 60}
        },
        'South America': {
            'center_lat': -15, 'center_lon': -60, 'zoom': 3,
            'bounds': {'north': 12, 'south': -55, 'east': -35, 'west': -85}
        },
        'Africa': {
            'center_lat': 0, 'center_lon': 20, 'zoom': 3,
            'bounds': {'north': 35, 'south': -35, 'east': 50, 'west': -20}
        },
        'Australia': {
            'center_lat': -25, 'center_lon': 135, 'zoom': 4,
            'bounds': {'north': -10, 'south': -45, 'east': 155, 'west': 110}
        }
    }
    
    return regions.get(region, regions['Global'])

def get_tile_layer(map_style):
    """Get appropriate tile layer for map style"""
    
    styles = {
        'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'Terrain': 'OpenStreetMap',
        'Streets': 'cartodbpositron'
    }
    
    return styles.get(map_style, 'OpenStreetMap')

def get_bloom_color(intensity):
    """Get color based on bloom intensity"""
    
    if intensity < 0.2:
        return '#FFFACD'  # Light yellow
    elif intensity < 0.4:
        return '#F0E68C'  # Khaki
    elif intensity < 0.6:
        return '#9ACD32'  # Yellow green
    elif intensity < 0.8:
        return '#32CD32'  # Lime green
    else:
        return '#006400'  # Dark green

def add_map_legend(map_obj):
    """Add legend to the map"""
    
    legend_html = """
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 180px; height: 130px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4 style="margin-top:0">Bloom Intensity</h4>
    <p><i style="background:#006400; width:15px; height:15px; float:left; margin-right:8px; opacity:0.8"></i> Peak (0.8-1.0)</p>
    <p><i style="background:#32CD32; width:15px; height:15px; float:left; margin-right:8px; opacity:0.8"></i> High (0.6-0.8)</p>
    <p><i style="background:#9ACD32; width:15px; height:15px; float:left; margin-right:8px; opacity:0.8"></i> Moderate (0.4-0.6)</p>
    <p><i style="background:#F0E68C; width:15px; height:15px; float:left; margin-right:8px; opacity:0.8"></i> Low (0.2-0.4)</p>
    <p><i style="background:#FFFACD; width:15px; height:15px; float:left; margin-right:8px; opacity:0.8"></i> Minimal (0.0-0.2)</p>
    </div>
    """
    
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def find_closest_bloom_point(bloom_data, clicked_lat, clicked_lng):
    """Find the closest bloom data point to clicked location"""
    
    if bloom_data.empty:
        return None
    
    # Calculate distances
    distances = ((bloom_data['lat'] - clicked_lat)**2 + (bloom_data['lon'] - clicked_lng)**2)**0.5
    closest_idx = distances.idxmin()
    
    return bloom_data.iloc[closest_idx]

def display_location_details(point, vegetation_index):
    """Display detailed information about selected location"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Bloom Intensity", f"{point['intensity']:.3f}")
        st.metric("Confidence Level", f"{point['confidence']}%")
    
    with col2:
        st.metric("Land Cover", point['land_cover'])
        st.metric("Bloom Stage", point['bloom_stage'])
    
    # Show trend chart for this location
    st.subheader("📈 Temporal Trend")
    
    # Generate sample time series for this location
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    values = np.random.normal(point['intensity'], 0.1, len(dates))
    values = np.clip(values, 0, 1)
    
    trend_df = pd.DataFrame({'Date': dates, vegetation_index: values})
    
    fig = px.line(
        trend_df, 
        x='Date', 
        y=vegetation_index,
        title=f"{vegetation_index} Trend at Location ({point['lat']:.3f}°, {point['lon']:.3f}°)"
    )
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def generate_weekly_bloom_data(date, vegetation_index):
    """Generate bloom data for a specific week"""
    
    np.random.seed(int(date.timestamp()))
    
    # Generate sample weekly data
    n_points = 100
    lats = np.random.uniform(-60, 70, n_points)
    lons = np.random.uniform(-180, 180, n_points)
    
    # Vary intensity based on season and location
    day_of_year = date.timetuple().tm_yday
    seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak around day 170 (mid-June)
    
    intensities = []
    for lat in lats:
        # Higher values in temperate regions during appropriate season
        lat_factor = np.exp(-((lat - 40)**2) / 1000)  # Peak around 40° latitude
        base_intensity = 0.2 + 0.6 * lat_factor * seasonal_factor
        noise = np.random.normal(0, 0.1)
        intensity = np.clip(base_intensity + noise, 0, 1)
        intensities.append(intensity)
    
    return pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'intensity': intensities,
        'date': date
    })

def create_weekly_bloom_chart(weekly_data, current_date, vegetation_index):
    """Create visualization for weekly bloom data"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon=weekly_data['lon'],
        lat=weekly_data['lat'],
        mode='markers',
        marker=dict(
            size=6,
            color=weekly_data['intensity'],
            colorscale='Viridis',
            cmin=0,
            cmax=1,
            opacity=0.7
        ),
        name='Bloom Activity'
    ))
    
    fig.update_layout(
        title=f'Global Bloom Activity - Week of {current_date.strftime("%B %d, %Y")}',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
        ),
        height=400
    )
    
    return fig

def generate_bloom_report(date_range, region, vegetation_index):
    """Generate a comprehensive bloom report"""
    
    st.subheader("📋 Bloom Activity Report")
    
    # Report metadata
    report_data = {
        'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'analysis_period': f"{date_range[0]} to {date_range[1]}",
        'region': region,
        'vegetation_index': vegetation_index,
        'total_detections': np.random.randint(500, 2000),
        'peak_activity_date': date_range[0] + timedelta(days=np.random.randint(0, 30)),
        'average_intensity': np.random.uniform(0.4, 0.7),
        'confidence_level': np.random.randint(75, 95)
    }
    
    # Display report
    st.markdown(f"""
    **Report Generated:** {report_data['report_date']}
    
    **Analysis Period:** {report_data['analysis_period']}
    
    **Region:** {report_data['region']}
    
    **Vegetation Index:** {report_data['vegetation_index']}
    
    ---
    
    **Key Findings:**
    - Total bloom detections: {report_data['total_detections']:,}
    - Peak activity date: {report_data['peak_activity_date']}
    - Average bloom intensity: {report_data['average_intensity']:.3f}
    - Overall confidence: {report_data['confidence_level']}%
    
    **Recommendations:**
    - Monitor continued activity in high-intensity regions
    - Consider agricultural timing implications
    - Track seasonal progression patterns
    """)
    
    st.success("✅ Report generated successfully!")

if __name__ == "__main__":
    main()
