"""
NASA Data Discovery Page for BloomSphere
Browse and download satellite imagery from NASA Earthdata
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import folium
from streamlit_folium import folium_static
from utils.nasa_earthdata import NASAEarthdataClient
from utils.ui_components import (
    page_header, section_header, info_panel, COLORS
)

st.set_page_config(
    page_title="NASA Data Discovery - BloomSphere",
    page_icon="🛰️",
    layout="wide"
)

page_header(
    "NASA Satellite Data Discovery",
    "Browse and access satellite imagery from NASA's Earthdata system including MODIS, Landsat, and VIIRS datasets",
    icon="🛰️"
)

# Initialize NASA client
@st.cache_resource
def get_nasa_client():
    """Get or create NASA Earthdata client"""
    return NASAEarthdataClient()

nasa_client = get_nasa_client()

# Sidebar for search parameters
st.sidebar.markdown(f"""
    <h2 style="color: {COLORS['primary']}; margin-bottom: 1.5rem;">
        🔍 Search Parameters
    </h2>
""", unsafe_allow_html=True)

# Data source selection
data_source = st.sidebar.selectbox(
    "Satellite Data Source",
    options=['MODIS Terra', 'MODIS Aqua', 'Landsat 9', 'Landsat 8', 'VIIRS'],
    help="Select the satellite data source to search"
)

# Region selection
region_preset = st.sidebar.selectbox(
    "Select Region Preset",
    options=['Custom', 'California', 'Great Plains', 'Amazon', 'Sahel', 'Australia'],
    help="Choose a predefined region or enter custom coordinates"
)

# Get region info
if region_preset != 'Custom':
    region_info = nasa_client.get_sample_data_info(region_preset.lower().replace(' ', '_'))
    bbox = region_info['bbox']
    st.sidebar.success(f"📍 {region_info['name']}: {region_info['description']}")
else:
    st.sidebar.subheader("Custom Bounding Box")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        west = st.number_input("West", value=-120.0, min_value=-180.0, max_value=180.0)
        south = st.number_input("South", value=35.0, min_value=-90.0, max_value=90.0)
    with col2:
        east = st.number_input("East", value=-115.0, min_value=-180.0, max_value=180.0)
        north = st.number_input("North", value=40.0, min_value=-90.0, max_value=90.0)
    
    bbox = (west, south, east, north)

# Date range selection
st.sidebar.subheader("📅 Date Range")

date_preset = st.sidebar.selectbox(
    "Date Preset",
    options=['Custom', 'Last 30 days', 'Last 90 days', 'Last 6 months', 'Last year']
)

if date_preset == 'Custom':
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=90),
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
else:
    days_map = {
        'Last 30 days': 30,
        'Last 90 days': 90,
        'Last 6 months': 180,
        'Last year': 365
    }
    days = days_map[date_preset]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

# Convert to strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Search button
search_button = st.sidebar.button("🔎 Search Available Data", type="primary")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Search Results", "📊 Data Coverage", "📥 Download Queue", "ℹ️ About Data Sources"])

with tab1:
    st.header("Search Results")
    
    if search_button or 'granules' not in st.session_state:
        with st.spinner("Searching NASA Earthdata..."):
            
            # Map data source to search method
            if data_source.startswith('MODIS'):
                satellite = 'terra' if 'Terra' in data_source else 'aqua'
                granules = nasa_client.get_modis_vegetation_indices(
                    bbox, start_date_str, end_date_str, satellite
                )
            elif data_source.startswith('Landsat'):
                version = int(data_source.split()[-1])
                granules = nasa_client.get_landsat_imagery(
                    bbox, start_date_str, end_date_str, version
                )
            elif data_source == 'VIIRS':
                granules = nasa_client.get_viirs_vegetation(
                    bbox, start_date_str, end_date_str
                )
            else:
                granules = []
            
            # If no real granules found, use mock data for demonstration
            if not granules:
                st.info("💡 Using demonstration data (NASA Earthdata API requires authentication)")
                granules = nasa_client.create_mock_granule_list(bbox, num_granules=10)
            
            st.session_state.granules = granules
    
    # Display results
    if 'granules' in st.session_state and st.session_state.granules:
        granules = st.session_state.granules
        
        st.success(f"✅ Found {len(granules)} available granules")
        
        # Create DataFrame for display
        granule_data = []
        for i, g in enumerate(granules):
            granule_data.append({
                'Index': i,
                'Title': g.get('title', 'N/A'),
                'Date': g.get('time_start', 'N/A')[:10],
                'Collection': g.get('collection', 'N/A'),
                'Mock Data': '🔶 Demo' if g.get('mock_data') else '✅ Real',
                'ID': g.get('id', 'N/A')
            })
        
        df = pd.DataFrame(granule_data)
        
        # Display interactive table
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                'Index': st.column_config.NumberColumn('No.', width='small'),
                'Title': st.column_config.TextColumn('Granule Title', width='large'),
                'Date': st.column_config.DateColumn('Acquisition Date', width='medium'),
                'Collection': st.column_config.TextColumn('Collection', width='medium'),
                'Mock Data': st.column_config.TextColumn('Status', width='small')
            },
            hide_index=True
        )
        
        # Granule details
        st.subheader("📋 Granule Details")
        
        selected_idx = st.selectbox(
            "Select a granule to view details",
            options=range(len(granules)),
            format_func=lambda i: f"{granules[i].get('title', f'Granule {i}')}"
        )
        
        if selected_idx is not None:
            selected_granule = granules[selected_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.json(selected_granule)
            
            with col2:
                st.markdown("### Actions")
                
                if st.button("📥 Add to Download Queue"):
                    if 'download_queue' not in st.session_state:
                        st.session_state.download_queue = []
                    
                    if selected_granule not in st.session_state.download_queue:
                        st.session_state.download_queue.append(selected_granule)
                        st.success("Added to download queue!")
                    else:
                        st.info("Already in queue")
                
                if selected_granule.get('mock_data'):
                    st.info("🔶 This is demonstration data. Real data requires NASA Earthdata credentials.")
                else:
                    st.success("✅ Real data available for download")
    
    else:
        st.info("👆 Click 'Search Available Data' to find satellite imagery")
        
        # Show search map
        st.subheader("🗺️ Search Region")
        m = folium.Map(
            location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
            zoom_start=5
        )
        
        # Add bounding box
        folium.Rectangle(
            bounds=[(bbox[1], bbox[0]), (bbox[3], bbox[2])],
            color='#3388ff',
            fill=True,
            fillOpacity=0.2,
            popup=f"Search Area: {region_preset if region_preset != 'Custom' else 'Custom Region'}"
        ).add_to(m)
        
        folium_static(m, width=800, height=400)

with tab2:
    st.header("📊 Data Coverage Analysis")
    
    if 'granules' in st.session_state and st.session_state.granules:
        granules = st.session_state.granules
        
        # Extract dates
        dates = []
        for g in granules:
            if g.get('time_start'):
                try:
                    date_obj = datetime.fromisoformat(g['time_start'].replace('Z', '+00:00'))
                    dates.append(date_obj)
                except:
                    pass
        
        if dates:
            # Create coverage DataFrame
            coverage_df = pd.DataFrame({
                'Date': dates,
                'Granules': [1] * len(dates)
            })
            
            coverage_df = coverage_df.groupby(pd.Grouper(key='Date', freq='D')).sum().reset_index()
            
            # Display timeline
            st.subheader("📅 Temporal Coverage")
            st.line_chart(coverage_df.set_index('Date')['Granules'])
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Granules", len(dates))
            
            with col2:
                date_range = (max(dates) - min(dates)).days
                st.metric("Date Range (days)", date_range)
            
            with col3:
                avg_coverage = len(dates) / max(date_range, 1)
                st.metric("Avg. Granules/Day", f"{avg_coverage:.2f}")
            
            # Calendar view
            st.subheader("📆 Available Data Calendar")
            
            date_list = [d.date() for d in dates]
            unique_dates = sorted(set(date_list))
            
            st.write(f"Data available on **{len(unique_dates)}** unique days")
            
            # Group by month
            month_counts = {}
            for d in dates:
                month_key = d.strftime('%Y-%m')
                month_counts[month_key] = month_counts.get(month_key, 0) + 1
            
            month_df = pd.DataFrame([
                {'Month': k, 'Granules': v} 
                for k, v in sorted(month_counts.items())
            ])
            
            st.bar_chart(month_df.set_index('Month')['Granules'])
    else:
        st.info("No data to analyze. Please search for granules first.")

with tab3:
    st.header("📥 Download Queue")
    
    if 'download_queue' in st.session_state and st.session_state.download_queue:
        st.success(f"**{len(st.session_state.download_queue)}** items in queue")
        
        # Display queue
        queue_data = []
        for i, g in enumerate(st.session_state.download_queue):
            queue_data.append({
                'No.': i + 1,
                'Title': g.get('title', 'N/A'),
                'Date': g.get('time_start', 'N/A')[:10],
                'Collection': g.get('collection', 'N/A'),
                'Type': 'Demo' if g.get('mock_data') else 'Real'
            })
        
        queue_df = pd.DataFrame(queue_data)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Queue"):
                st.session_state.download_queue = []
                st.rerun()
        
        with col2:
            if st.button("⬇️ Download All"):
                st.info("Download functionality requires NASA Earthdata credentials. Mock data can be generated for demonstration.")
        
        with col3:
            if st.button("📋 Export Queue List"):
                queue_json = pd.DataFrame(st.session_state.download_queue).to_json(orient='records', indent=2)
                st.download_button(
                    label="Download Queue as JSON",
                    data=queue_json,
                    file_name=f"download_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("📭 Queue is empty. Add granules from the search results.")

with tab4:
    st.header("ℹ️ About NASA Satellite Data Sources")
    
    st.markdown("""
    ### 🛰️ Available Data Sources
    
    #### MODIS (Moderate Resolution Imaging Spectroradiometer)
    - **Satellites**: Terra (AM) and Aqua (PM)
    - **Temporal Resolution**: Daily, with composite products every 8 or 16 days
    - **Spatial Resolution**: 250m, 500m, 1km (depending on product)
    - **Key Products**: 
      - MOD13A2/MYD13A2: 16-day NDVI/EVI at 1km
      - MOD09A1/MYD09A1: 8-day Surface Reflectance at 500m
    - **Best For**: Large-scale vegetation monitoring, global bloom patterns
    
    #### Landsat
    - **Satellites**: Landsat 8 and Landsat 9
    - **Temporal Resolution**: 16 days (8 days combined)
    - **Spatial Resolution**: 30m (multispectral), 15m (panchromatic)
    - **Key Products**: Level-2 Surface Reflectance
    - **Best For**: Regional and local bloom analysis, high-detail mapping
    
    #### VIIRS (Visible Infrared Imaging Radiometer Suite)
    - **Satellite**: Suomi NPP, NOAA-20
    - **Temporal Resolution**: Daily
    - **Spatial Resolution**: 375m - 750m
    - **Key Products**: VNP13A1 (16-day vegetation indices)
    - **Best For**: Frequent monitoring, continuity with MODIS
    
    ### 🔑 Authentication
    
    To access real NASA Earthdata, you need:
    1. **Earthdata Login Account** - Register at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)
    2. **API Credentials** - Set environment variables:
       - `NASA_EARTHDATA_USERNAME`
       - `NASA_EARTHDATA_PASSWORD`
    
    ### 📚 Resources
    - [NASA Earthdata](https://earthdata.nasa.gov/)
    - [MODIS Vegetation Index Products](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php)
    - [Landsat Science](https://landsat.gsfc.nasa.gov/)
    - [VIIRS Vegetation Products](https://viirsland.gsfc.nasa.gov/)
    """)
    
    # Connection test
    st.subheader("🔌 Connection Test")
    
    if st.button("Test NASA Earthdata Connection"):
        with st.spinner("Testing connection..."):
            test_results = nasa_client.test_connection()
            
            if test_results['cmr_accessible']:
                st.success("✅ CMR Search API is accessible")
            else:
                st.error("❌ Cannot reach CMR Search API")
            
            if test_results.get('error'):
                st.warning(f"⚠️ Error: {test_results['error']}")
            
            if not nasa_client.username or not nasa_client.password:
                st.info("ℹ️ No credentials configured. Add NASA_EARTHDATA_USERNAME and NASA_EARTHDATA_PASSWORD to enable authenticated downloads.")
            elif test_results['authenticated']:
                st.success("✅ Authentication credentials detected")
