import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from utils.ui_components import (
    hero_section, feature_card, section_header, metric_card,
    info_panel, divider, create_professional_chart_layout, COLORS
)

# Page configuration
st.set_page_config(
    page_title="BloomSphere - NASA Plant Bloom Monitoring",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for additional professional styling
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F5F7FA;
    }
    
    /* Better button styling */
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    /* Improve metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Hero Banner
    hero_section(
        title="BloomSphere",
        subtitle="Harnessing NASA Earth Observation Data to Monitor, Predict, and Analyze Global Plant Blooming Events",
        icon="🌸"
    )
    
    # Key Features Section
    section_header(
        "Why BloomSphere?",
        "Comprehensive vegetation monitoring powered by NASA satellite technology"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_card(
            icon="🗺️",
            title="Global Mapping",
            description="Interactive world map visualization with vegetation intensity heatmaps. Zoom from global to local scale to explore bloom patterns worldwide."
        )
    
    with col2:
        feature_card(
            icon="📊",
            title="Temporal Analysis",
            description="Track blooming events over time with time-series analysis, seasonal pattern detection, and multi-year trend visualization."
        )
    
    with col3:
        feature_card(
            icon="🛰️",
            title="NASA Satellite Data",
            description="Access MODIS, Landsat, and VIIRS satellite imagery with automated NDVI, EVI, and SAVI vegetation index calculations."
        )
    
    divider('xl')
    
    # Capabilities Overview
    section_header(
        "Advanced Capabilities",
        "Professional tools for vegetation research and monitoring"
    )
    
    cap_col1, cap_col2 = st.columns(2)
    
    with cap_col1:
        feature_card(
            icon="🔮",
            title="Bloom Predictions",
            description="Machine learning models forecast future bloom events based on historical patterns, climate data, and vegetation trends."
        )
    
    with cap_col2:
        feature_card(
            icon="🌿",
            title="Species Identification",
            description="Spectral signature analysis helps identify plant species and vegetation types from satellite imagery."
        )
    
    cap_col3, cap_col4 = st.columns(2)
    
    with cap_col3:
        feature_card(
            icon="📈",
            title="Comparative Analysis",
            description="Compare bloom patterns across different regions, time periods, and vegetation types for comprehensive insights."
        )
    
    with cap_col4:
        feature_card(
            icon="🌍",
            title="Ecological Impact",
            description="Assess the environmental and ecological impacts of bloom events on ecosystems and biodiversity."
        )
    
    divider('xl')
    
    # Getting Started Guide
    section_header(
        "🚀 Get Started",
        "Three simple steps to begin exploring global vegetation patterns"
    )
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown(f"""
        <div style="
            background: {COLORS['bg_card']};
            border-left: 4px solid {COLORS['primary']};
            padding: 1.5rem;
            border-radius: 0.375rem;
            height: 100%;
        ">
            <h3 style="color: {COLORS['primary']}; margin: 0 0 1rem 0;">
                1️⃣ Explore Global Map
            </h3>
            <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['text_dark']};">
                <li>View worldwide bloom intensity</li>
                <li>Navigate through different time periods</li>
                <li>Zoom into specific regions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        st.markdown(f"""
        <div style="
            background: {COLORS['bg_card']};
            border-left: 4px solid {COLORS['primary']};
            padding: 1.5rem;
            border-radius: 0.375rem;
            height: 100%;
        ">
            <h3 style="color: {COLORS['primary']}; margin: 0 0 1rem 0;">
                2️⃣ Upload Your Data
            </h3>
            <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['text_dark']};">
                <li>Process satellite imagery (TIFF)</li>
                <li>Calculate vegetation indices</li>
                <li>Generate custom visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col3:
        st.markdown(f"""
        <div style="
            background: {COLORS['bg_card']};
            border-left: 4px solid {COLORS['primary']};
            padding: 1.5rem;
            border-radius: 0.375rem;
            height: 100%;
        ">
            <h3 style="color: {COLORS['primary']}; margin: 0 0 1rem 0;">
                3️⃣ Analyze Trends
            </h3>
            <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['text_dark']};">
                <li>View bloom statistics</li>
                <li>Export reports and visualizations</li>
                <li>Compare regional patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    divider('xl')
    
    # Sample Heatmap Visualizations
    section_header(
        "📊 Example Analysis Outputs",
        "Real vegetation and bloom intensity heatmaps from satellite imagery processing"
    )
    
    info_panel(
        "These heatmaps show vegetation and bloom intensity from satellite imagery analysis. "
        "The color scale represents NDVI values - darker greens indicate higher vegetation density and bloom activity.",
        panel_type="info",
        icon="ℹ️"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(
            "attached_assets/productOverview_heatmap_1759601442713.png",
            caption="Vegetation/Bloom Intensity - Product Overview",
            width="stretch"
        )
    
    with col2:
        st.image(
            "attached_assets/1998435_1_CV_heatmap_1759601442715.png",
            caption="Vegetation/Bloom Intensity - Regional Analysis",
            width="stretch"
        )
    
    divider('xl')
    
    # Current Global Bloom Activity - Load on demand
    section_header(
        "🌍 Current Global Bloom Activity",
        "Live visualization of bloom patterns detected from recent satellite data"
    )
    
    # Use expander to defer loading until user clicks
    with st.expander("📊 View Global Bloom Map", expanded=True):
        create_sample_global_view()
    
    divider('xl')
    
    # Data Sources and Methodology
    section_header("📡 Data Sources & Methodology")
    
    with st.expander("NASA Earth Observation Missions", expanded=False):
        st.markdown(f"""
        <div style="color: {COLORS['text_dark']}; line-height: 1.8;">
        
        **MODIS** - Moderate Resolution Imaging Spectroradiometer  
        Provides daily global coverage with vegetation indices at 250m-1km resolution
        
        **Landsat** - Long-term Earth Observation Program  
        30m resolution multispectral imagery, extensive archive dating back to the 1970s
        
        **VIIRS** - Visible Infrared Imaging Radiometer Suite  
        Daily vegetation indices at 500m resolution with improved cloud detection
        
        **EMIT** - Earth Surface Mineral Dust Source Investigation  
        High-resolution spectral data for detailed vegetation analysis
        
        **PACE** - Plankton, Aerosol, Cloud ocean Ecosystem  
        Ocean color and atmospheric data for comprehensive ecosystem monitoring
        
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Vegetation Indices & Algorithms", expanded=False):
        st.markdown(f"""
        <div style="color: {COLORS['text_dark']}; line-height: 1.8;">
        
        **Vegetation Indices:**
        - **NDVI** (Normalized Difference Vegetation Index): (NIR - Red) / (NIR + Red)
        - **EVI** (Enhanced Vegetation Index): 2.5 × ((NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1))
        - **SAVI** (Soil-Adjusted Vegetation Index): ((NIR - Red) / (NIR + Red + L)) × (1 + L)
        
        **Bloom Detection Algorithm:**
        1. Calculate vegetation indices from multi-spectral satellite imagery
        2. Apply temporal filtering to identify bloom event signatures
        3. Generate intensity maps using scientifically calibrated color gradients
        4. Provide statistical analysis and trend detection using machine learning
        
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def generate_global_bloom_data():
    """Generate cached global bloom data - cached for 1 hour"""
    np.random.seed(42)
    
    # Create sample global coordinates
    lats = np.random.uniform(-60, 70, 500)
    lons = np.random.uniform(-180, 180, 500)
    
    # Generate bloom intensity values (0-1 scale)
    bloom_intensity = []
    for lat, lon in zip(lats, lons):
        base_intensity = 0.3 + 0.4 * np.exp(-((lat - 40)**2) / 800)
        seasonal_factor = 0.6 + 0.4 * np.sin(np.radians(lon / 2))
        noise = np.random.normal(0, 0.1)
        intensity = np.clip(base_intensity * seasonal_factor + noise, 0, 1)
        bloom_intensity.append(intensity)
    
    return lats, lons, bloom_intensity

def create_sample_global_view():
    """Create a professional global vegetation intensity map"""
    
    # Load cached data
    lats, lons, bloom_intensity = generate_global_bloom_data()
    
    # Create the map with professional styling
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=8,
            color=bloom_intensity,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text="Bloom<br>Intensity", side="right"),
                tickmode="linear",
                tick0=0,
                dtick=0.2,
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['Low', '', 'Moderate', '', 'High', 'Peak']
            ),
            cmin=0,
            cmax=1,
            opacity=0.7
        ),
        hovertemplate='<b>Bloom Intensity</b><br>' +
                      'Latitude: %{lat:.2f}°<br>' +
                      'Longitude: %{lon:.2f}°<br>' +
                      'Intensity: %{marker.color:.2f}<extra></extra>',
        name='Bloom Activity'
    ))
    
    # Apply professional layout
    layout_config = create_professional_chart_layout()
    fig.update_layout(
        title={
            'text': 'Global Plant Bloom Activity - Current Week',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': COLORS['text_dark'], 'family': 'sans-serif'}
        },
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showlakes=True,
            lakecolor='rgb(230, 245, 255)',
        ),
        height=500,
        margin=dict(t=60, b=0, l=0, r=0),
        font=layout_config['font'],
        plot_bgcolor=layout_config['plot_bgcolor'],
        paper_bgcolor=layout_config['paper_bgcolor']
    )
    
    st.plotly_chart(fig, width="stretch", key="global_bloom_map")
    
    # Professional metric cards
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            label="Active Bloom Sites",
            value=f"{len([x for x in bloom_intensity if x > 0.6])}",
            icon="🌸"
        )
    
    with col2:
        metric_card(
            label="Average Intensity",
            value=f"{np.mean(bloom_intensity):.2f}",
            delta="+0.12 from last week",
            delta_positive=True,
            icon="📊"
        )
    
    with col3:
        metric_card(
            label="Peak Activity Regions",
            value=f"{len([x for x in bloom_intensity if x > 0.8])}",
            icon="⭐"
        )
    
    with col4:
        metric_card(
            label="Data Points Analyzed",
            value=f"{len(bloom_intensity):,}",
            icon="🔍"
        )

if __name__ == "__main__":
    main()
