"""
Species Analysis and Identification Page for BloomSphere
Spectral signature analysis for vegetation type identification
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.species_identifier import SpeciesIdentifier
from utils.ui_components import (
    page_header, section_header, info_panel, COLORS
)

st.set_page_config(
    page_title="Species Analysis - BloomSphere",
    page_icon="🌿",
    layout="wide"
)

page_header(
    "Vegetation Species Analysis",
    "Identify plant species and vegetation types using spectral signature analysis. Our system analyzes vegetation indices and reflectance patterns to suggest likely species.",
    icon="🌿"
)

# Initialize species identifier
@st.cache_resource
def get_identifier():
    """Get or create species identifier"""
    return SpeciesIdentifier()

identifier = get_identifier()

# Tabs for different analysis features
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Species Identification",
    "📊 Spectral Analysis",
    "📚 Species Database",
    "🗺️ Vegetation Mapping"
])

with tab1:
    section_header("Species Identification from Vegetation Indices")
    
    st.markdown("""
    Enter vegetation index values from your satellite imagery to identify likely plant species.
    """)
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        options=["Manual Entry", "Sample Data", "From Processed Image"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Required Indices")
            ndvi_input = st.slider(
                "NDVI (Normalized Difference Vegetation Index)",
                min_value=-1.0,
                max_value=1.0,
                value=0.6,
                step=0.01,
                help="Typical vegetation: 0.3-0.8. Higher = more vegetation"
            )
            
            evi_input = st.slider(
                "EVI (Enhanced Vegetation Index)",
                min_value=-1.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Improved sensitivity in high biomass regions"
            )
        
        with col2:
            st.subheader("Optional Reflectances")
            include_reflectance = st.checkbox("Include band reflectance data")
            
            if include_reflectance:
                red_input = st.slider(
                    "Red Band Reflectance",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.10,
                    step=0.01
                )
                
                nir_input = st.slider(
                    "NIR Band Reflectance",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.35,
                    step=0.01
                )
            else:
                red_input = None
                nir_input = None
    
    elif input_method == "Sample Data":
        st.subheader("Select Sample Vegetation Type")
        
        sample_types = {
            "Healthy Corn Field": (0.75, 0.65, 0.10, 0.42),
            "Wheat Field": (0.65, 0.55, 0.12, 0.38),
            "Deciduous Forest (Spring)": (0.70, 0.55, 0.08, 0.40),
            "Grassland": (0.45, 0.35, 0.15, 0.30),
            "Wildflower Meadow": (0.60, 0.50, 0.12, 0.36),
            "Desert Bloom": (0.35, 0.28, 0.22, 0.28),
            "Fruit Orchard (Bloom)": (0.68, 0.58, 0.10, 0.39)
        }
        
        selected_sample = st.selectbox(
            "Choose a sample type",
            options=list(sample_types.keys())
        )
        
        ndvi_input, evi_input, red_input, nir_input = sample_types[selected_sample]
        
        st.info(f"📊 **{selected_sample}**: NDVI={ndvi_input:.2f}, EVI={evi_input:.2f}")
    
    else:  # From Processed Image
        st.info("💡 This feature would connect to previously processed imagery. Using sample data for demonstration.")
        ndvi_input, evi_input, red_input, nir_input = 0.65, 0.55, 0.11, 0.38
    
    # Identify button
    if st.button("🔍 Identify Species", type="primary"):
        with st.spinner("Analyzing spectral signature..."):
            # Perform identification
            matches = identifier.identify_species(
                ndvi=ndvi_input,
                evi=evi_input,
                red=red_input,
                nir=nir_input,
                top_n=5
            )
            
            # Get vegetation categorization
            veg_category = identifier.categorize_vegetation(ndvi_input, evi_input)
        
        # Display results
        st.success("✅ Analysis Complete!")
        
        # Vegetation health overview
        st.subheader("🌱 Vegetation Health Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Health Status", veg_category['health_status'])
        
        with col2:
            st.metric("Vigor", veg_category['vigor'])
        
        with col3:
            st.metric("Density", veg_category['density'])
        
        with col4:
            st.metric("Activity", veg_category['photosynthetic_activity'])
        
        # Species matches
        st.subheader("🎯 Top Species Matches")
        
        for i, match in enumerate(matches, 1):
            with st.expander(
                f"#{i}: {match['species'].replace('_', ' ').title()} - {match['confidence']:.1f}% confidence",
                expanded=(i == 1)
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Category:** {match['category']}")
                    st.markdown(f"**Bloom Season:** {match['bloom_season']}")
                    st.markdown(f"**Description:** {match['description']}")
                
                with col2:
                    # Confidence gauge
                    fig_confidence = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=match['confidence'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Match %"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#2ecc71"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                            ],
                        }
                    ))
                    fig_confidence.update_layout(height=200, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_confidence, use_container_width=True, key=f"confidence_gauge_{i}")
                
                # Show expected ranges
                st.markdown("**Expected Spectral Ranges:**")
                st.write(f"- NDVI: {match['ndvi_range'][0]:.2f} - {match['ndvi_range'][1]:.2f}")
                st.write(f"- EVI: {match['evi_range'][0]:.2f} - {match['evi_range'][1]:.2f}")
        
        # Comparison chart
        st.subheader("📊 Spectral Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for match in matches[:3]:
            comparison_data.append({
                'Species': match['species'].replace('_', ' ').title(),
                'Your NDVI': ndvi_input,
                'Expected NDVI': np.mean(match['ndvi_range']),
                'Your EVI': evi_input,
                'Expected EVI': np.mean(match['evi_range'])
            })
        
        if comparison_data:
            df_compare = pd.DataFrame(comparison_data)
            
            fig_compare = go.Figure()
            
            # NDVI comparison
            fig_compare.add_trace(go.Bar(
                name='Your NDVI',
                x=df_compare['Species'],
                y=df_compare['Your NDVI'],
                marker_color='#3498db'
            ))
            
            fig_compare.add_trace(go.Bar(
                name='Expected NDVI',
                x=df_compare['Species'],
                y=df_compare['Expected NDVI'],
                marker_color='#2ecc71'
            ))
            
            fig_compare.update_layout(
                title="NDVI Comparison with Expected Values",
                xaxis_title="Species",
                yaxis_title="NDVI",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_compare, use_container_width=True, key="species_comparison_chart")

with tab2:
    st.header("Comprehensive Spectral Analysis")
    
    st.markdown("""
    Detailed analysis of spectral signatures from multi-band satellite imagery.
    """)
    
    # Band reflectance inputs
    st.subheader("📡 Input Band Reflectances")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        blue_band = st.number_input("Blue Band", 0.0, 1.0, 0.08, 0.01)
    
    with col2:
        green_band = st.number_input("Green Band", 0.0, 1.0, 0.12, 0.01)
    
    with col3:
        red_band = st.number_input("Red Band", 0.0, 1.0, 0.10, 0.01)
    
    with col4:
        nir_band = st.number_input("NIR Band", 0.0, 1.0, 0.38, 0.01)
    
    if st.button("Analyze Spectral Signature"):
        # Perform comprehensive analysis
        analysis = identifier.analyze_spectral_signature(
            red=red_band,
            green=green_band,
            blue=blue_band,
            nir=nir_band
        )
        
        # Display vegetation indices
        st.subheader("🌿 Calculated Vegetation Indices")
        
        indices = analysis['vegetation_indices']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("NDVI", f"{indices['NDVI']:.3f}")
        
        with col2:
            st.metric("EVI", f"{indices['EVI']:.3f}")
        
        with col3:
            st.metric("SAVI", f"{indices['SAVI']:.3f}")
        
        with col4:
            st.metric("Green NDVI", f"{indices['Green_NDVI']:.3f}")
        
        # Spectral properties
        st.subheader("📊 Spectral Properties")
        
        props = analysis['spectral_properties']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Red Edge Position:** {props['red_edge_position']:.1f} nm")
            st.markdown(f"**Chlorophyll Index:** {props['chlorophyll_index']:.3f}")
        
        with col2:
            st.markdown(f"**Water Content Index:** {props['water_content_index']:.3f}")
            st.markdown(f"**NIR/Red Ratio:** {props['nir_red_ratio']:.3f}")
        
        # Spectral curve
        st.subheader("📈 Spectral Reflectance Curve")
        
        bands_data = analysis['band_reflectances']
        
        wavelengths = [450, 550, 650, 850]  # Approximate central wavelengths
        reflectances = [bands_data['blue'], bands_data['green'], 
                       bands_data['red'], bands_data['nir']]
        band_names = ['Blue', 'Green', 'Red', 'NIR']
        
        fig_spectrum = go.Figure()
        
        fig_spectrum.add_trace(go.Scatter(
            x=wavelengths,
            y=reflectances,
            mode='lines+markers',
            name='Reflectance',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ))
        
        # Add band labels
        for wl, refl, name in zip(wavelengths, reflectances, band_names):
            fig_spectrum.add_annotation(
                x=wl,
                y=refl,
                text=name,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363"
            )
        
        fig_spectrum.update_layout(
            title="Spectral Reflectance Signature",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            height=400
        )
        
        st.plotly_chart(fig_spectrum, use_container_width=True, key="spectral_reflectance_curve")
        
        # Interpretation
        st.subheader("🔬 Interpretation")
        
        if indices['NDVI'] > 0.6:
            st.success("✅ High vegetation vigor detected. Healthy, dense vegetation cover.")
        elif indices['NDVI'] > 0.3:
            st.info("ℹ️ Moderate vegetation detected. Typical for grasslands or crops in early growth.")
        else:
            st.warning("⚠️ Low vegetation signal. Sparse vegetation or stressed plants.")

with tab3:
    st.header("Species Reference Database")
    
    st.markdown("""
    Browse our reference database of vegetation spectral signatures.
    """)
    
    # Get all species
    all_species = identifier.get_all_species_info()
    
    # Category filter
    categories = list(set([s['category'] for s in all_species]))
    
    selected_category = st.selectbox(
        "Filter by Category",
        options=['All'] + sorted(categories)
    )
    
    if selected_category != 'All':
        filtered_species = [s for s in all_species if s['category'] == selected_category]
    else:
        filtered_species = all_species
    
    # Display species cards
    st.subheader(f"📚 {len(filtered_species)} Species")
    
    # Create DataFrame for table view
    species_df = pd.DataFrame([{
        'Name': s['name'].replace('_', ' ').title(),
        'Category': s['category'],
        'Bloom Season': s['bloom_season'],
        'NDVI Min': s['ndvi_range'][0],
        'NDVI Max': s['ndvi_range'][1],
        'EVI Min': s['evi_range'][0],
        'EVI Max': s['evi_range'][1],
        'Description': s['description']
    } for s in filtered_species])
    
    st.dataframe(
        species_df,
        use_container_width=True,
        column_config={
            'Name': st.column_config.TextColumn('Species/Type', width='medium'),
            'Category': st.column_config.TextColumn('Category', width='small'),
            'Bloom Season': st.column_config.TextColumn('Season', width='small'),
            'NDVI Min': st.column_config.NumberColumn('NDVI Min', format='%.2f'),
            'NDVI Max': st.column_config.NumberColumn('NDVI Max', format='%.2f'),
            'EVI Min': st.column_config.NumberColumn('EVI Min', format='%.2f'),
            'EVI Max': st.column_config.NumberColumn('EVI Max', format='%.2f'),
            'Description': st.column_config.TextColumn('Description', width='large')
        },
        hide_index=True
    )
    
    # Visualization of species ranges
    st.subheader("📊 Species NDVI Ranges")
    
    fig_ranges = go.Figure()
    
    for species in filtered_species[:10]:  # Show top 10
        name = species['name'].replace('_', ' ').title()
        ndvi_min, ndvi_max = species['ndvi_range']
        ndvi_avg = (ndvi_min + ndvi_max) / 2
        
        fig_ranges.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[ndvi_avg],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ndvi_max - ndvi_avg],
                arrayminus=[ndvi_avg - ndvi_min]
            ),
            marker_color='#2ecc71'
        ))
    
    fig_ranges.update_layout(
        title="NDVI Range Comparison (First 10 Species)",
        xaxis_title="Species",
        yaxis_title="NDVI",
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_ranges, use_container_width=True, key="species_ndvi_ranges")

with tab4:
    st.header("Vegetation Type Mapping")
    
    st.markdown("""
    Visualize vegetation type distribution based on spectral analysis.
    """)
    
    st.info("💡 This feature would integrate with processed satellite imagery to map vegetation types across a region. Sample visualization shown.")
    
    # Generate sample vegetation map data
    @st.cache_data
    def generate_veg_map_data():
        """Generate sample vegetation distribution data"""
        categories = ['Agricultural', 'Forest', 'Grassland', 'Shrubland', 'Wetland']
        colors = ['#f39c12', '#27ae60', '#95a5a6', '#d35400', '#3498db']
        
        data = []
        for i, cat in enumerate(categories):
            # Generate random points
            n_points = np.random.randint(20, 50)
            lats = np.random.uniform(35, 40, n_points)
            lons = np.random.uniform(-120, -115, n_points)
            
            for lat, lon in zip(lats, lons):
                data.append({
                    'Latitude': lat,
                    'Longitude': lon,
                    'Type': cat,
                    'Color': colors[i],
                    'NDVI': np.random.uniform(0.3, 0.8)
                })
        
        return pd.DataFrame(data)
    
    veg_map_df = generate_veg_map_data()
    
    # Display map
    fig_map = px.scatter_mapbox(
        veg_map_df,
        lat='Latitude',
        lon='Longitude',
        color='Type',
        size='NDVI',
        hover_data=['Type', 'NDVI'],
        zoom=6,
        height=600,
        title="Vegetation Type Distribution Map"
    )
    
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True, key="vegetation_map")
    
    # Summary statistics
    st.subheader("📊 Vegetation Type Distribution")
    
    type_counts = veg_map_df['Type'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.3
    )])
    
    fig_pie.update_layout(title="Distribution by Vegetation Type", height=400)
    st.plotly_chart(fig_pie, use_container_width=True, key="vegetation_distribution_pie")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About Species ID")
    
    st.markdown("""
    ### How It Works
    
    **Spectral Signatures**
    - Each plant species has unique reflectance patterns
    - Analyzed across visible and near-infrared bands
    - Compared against reference database
    
    **Vegetation Indices**
    - **NDVI**: Overall vegetation amount
    - **EVI**: Enhanced for dense vegetation
    - **SAVI**: Adjusted for soil background
    
    ### Accuracy Notes
    - Best results with clear, cloud-free imagery
    - Multiple observations improve accuracy
    - Consider seasonal variations
    - Ground truth validation recommended
    
    ### Database Coverage
    - Agricultural crops
    - Forest types
    - Grasslands and prairies
    - Flowering plants
    - Wetland vegetation
    """)
