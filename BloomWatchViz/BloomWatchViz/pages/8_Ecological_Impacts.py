"""
Ecological Impacts Page for BloomSphere
Analysis of bloom impacts on pollinators, agriculture, and ecosystem services
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from utils.ecological_impact import EcologicalImpactAnalyzer

st.set_page_config(
    page_title="Ecological Impacts - BloomSphere",
    page_icon="🦋",
    layout="wide"
)

st.title("🦋 Ecological Impact Analysis")
st.markdown("""
Understand the broader ecological and economic impacts of plant blooms including
pollinator activity, agricultural productivity, pollen forecasts, and ecosystem services.
""")

# Initialize impact analyzer
@st.cache_resource
def get_analyzer():
    """Get or create impact analyzer"""
    return EcologicalImpactAnalyzer()

analyzer = get_analyzer()

# Tabs for different impact analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🐝 Pollinator Activity",
    "🌾 Crop Yield Forecast",
    "🌼 Pollen Forecast",
    "💰 Ecosystem Services",
    "📄 Impact Report"
])

with tab1:
    st.header("Pollinator Activity Prediction")
    
    st.markdown("""
    Predict pollinator activity based on bloom characteristics and environmental conditions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bloom Characteristics")
        
        bloom_intensity_poll = st.slider(
            "Bloom Intensity (NDVI)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="Higher NDVI indicates more vigorous bloom"
        )
        
        vegetation_type_poll = st.selectbox(
            "Vegetation Type",
            options=['wildflowers', 'orchard', 'agricultural', 'forest', 'grassland', 'desert_bloom']
        )
    
    with col2:
        st.subheader("Environmental Conditions")
        
        temperature = st.slider(
            "Average Temperature (°C)",
            min_value=0.0,
            max_value=40.0,
            value=20.0,
            step=1.0,
            help="Optimal pollinator activity: 15-25°C"
        )
        
        day_of_year_poll = st.slider(
            "Day of Year",
            min_value=1,
            max_value=365,
            value=120,
            help="Day 1 = January 1, Day 120 ≈ late April"
        )
    
    if st.button("🔍 Predict Pollinator Activity", type="primary"):
        # Perform prediction
        pollinator_result = analyzer.predict_pollinator_activity(
            bloom_intensity=bloom_intensity_poll,
            vegetation_type=vegetation_type_poll,
            temperature=temperature,
            day_of_year=day_of_year_poll
        )
        
        # Display results
        st.success("✅ Analysis Complete!")
        
        # Activity metrics
        st.subheader("📊 Pollinator Activity Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Activity Score",
                f"{pollinator_result['activity_score']:.1f}/100",
                delta=pollinator_result['activity_level']
            )
        
        with col2:
            st.metric(
                "Foraging Area",
                f"{pollinator_result['estimated_foraging_area_km2']:.1f} km²"
            )
        
        with col3:
            # Create gauge chart for activity
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pollinator_result['activity_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Activity Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#f39c12"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "darkgray"},
                        {'range': [75, 100], 'color': "#2ecc71"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Description
        st.info(pollinator_result['description'])
        
        # Contributing factors
        st.subheader("🔍 Contributing Factors")
        
        factors_df = pd.DataFrame([
            {'Factor': 'Vegetation Type', 'Score': pollinator_result['vegetation_factor'] * 100, 'Impact': 'High' if pollinator_result['vegetation_factor'] > 0.8 else 'Moderate'},
            {'Factor': 'Temperature', 'Score': pollinator_result['temperature_factor'] * 100, 'Impact': 'High' if pollinator_result['temperature_factor'] > 0.8 else 'Moderate'},
            {'Factor': 'Season', 'Score': pollinator_result['seasonal_factor'] * 100, 'Impact': 'High' if pollinator_result['seasonal_factor'] > 0.8 else 'Moderate'}
        ])
        
        fig_factors = go.Figure(data=[go.Bar(
            x=factors_df['Factor'],
            y=factors_df['Score'],
            marker_color=['#3498db', '#e74c3c', '#2ecc71'],
            text=factors_df['Score'].round(1),
            textposition='outside'
        )])
        
        fig_factors.update_layout(
            title="Factor Contributions to Pollinator Activity",
            yaxis_title="Score (%)",
            height=400
        )
        
        st.plotly_chart(fig_factors, use_container_width=True)
        
        # Dominant pollinators
        st.subheader("🐝 Expected Pollinator Species")
        
        for pollinator in pollinator_result['dominant_pollinators']:
            st.markdown(f"- **{pollinator}**")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        for rec in pollinator_result['recommendations']:
            st.markdown(f"✓ {rec}")

with tab2:
    st.header("Agricultural Yield Forecast")
    
    st.markdown("""
    Forecast crop yield based on bloom characteristics and pollinator activity.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        bloom_coverage_yield = st.slider(
            "Bloom Coverage (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=5.0,
            help="Percentage of field in bloom"
        )
        
        bloom_intensity_yield = st.slider(
            "Bloom Intensity (NDVI)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05
        )
    
    with col2:
        crop_type = st.selectbox(
            "Crop Type",
            options=['orchard', 'berry', 'vegetables', 'wheat', 'corn', 'soybean'],
            help="Different crops have varying pollinator dependencies"
        )
        
        pollinator_activity_yield = st.slider(
            "Pollinator Activity Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=5.0,
            help="From pollinator activity analysis"
        )
    
    if st.button("📊 Forecast Crop Yield"):
        # Perform forecast
        yield_result = analyzer.forecast_crop_yield(
            bloom_coverage=bloom_coverage_yield,
            bloom_intensity=bloom_intensity_yield,
            crop_type=crop_type,
            pollinator_activity=pollinator_activity_yield
        )
        
        st.success("✅ Yield Forecast Complete!")
        
        # Yield metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Yield Outlook",
                yield_result['outlook']
            )
        
        with col2:
            st.metric(
                "Yield Index",
                f"{yield_result['yield_index']:.1f}/100"
            )
        
        with col3:
            st.metric(
                "Expected Yield",
                yield_result['expected_yield_pct']
            )
        
        # Yield composition
        st.subheader("📊 Yield Components")
        
        components = [
            {'Component': 'Bloom Quality', 'Contribution': yield_result['bloom_contribution']},
            {'Component': 'Pollinator Services', 'Contribution': yield_result['pollinator_contribution']}
        ]
        
        fig_yield_pie = go.Figure(data=[go.Pie(
            labels=[c['Component'] for c in components],
            values=[c['Contribution'] for c in components],
            hole=0.3
        )])
        
        fig_yield_pie.update_layout(
            title=f"Yield Index Components ({crop_type.title()})",
            height=400
        )
        
        st.plotly_chart(fig_yield_pie, use_container_width=True)
        
        # Pollinator dependency
        st.info(f"**Pollinator Dependency:** {yield_result['pollinator_dependency']*100:.0f}% - This crop's yield {'is highly dependent on' if yield_result['pollinator_dependency'] > 0.6 else 'has moderate dependency on' if yield_result['pollinator_dependency'] > 0.3 else 'has low dependency on'} pollinator services.")
        
        # Recommendations
        st.subheader("💡 Management Recommendations")
        
        for rec in yield_result['recommendations']:
            st.markdown(f"✓ {rec}")

with tab3:
    st.header("Pollen Forecast & Allergy Risk")
    
    st.markdown("""
    Estimate pollen production and allergic reaction risks based on bloom extent.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        vegetation_type_pollen = st.selectbox(
            "Vegetation Type",
            options=['grass', 'trees', 'wildflowers', 'agricultural', 'orchard'],
            key='pollen_veg_type',
            help="Different vegetation types produce different pollen levels"
        )
        
        bloom_area = st.number_input(
            "Bloom Coverage Area (km²)",
            min_value=0.1,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )
    
    with col2:
        bloom_intensity_pollen = st.slider(
            "Bloom Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key='pollen_intensity'
        )
        
        day_of_year_pollen = st.slider(
            "Day of Year",
            min_value=1,
            max_value=365,
            value=120,
            key='pollen_day',
            help="Pollen levels vary by season"
        )
    
    if st.button("🌼 Generate Pollen Forecast"):
        # Estimate pollen
        pollen_result = analyzer.estimate_pollen_production(
            vegetation_type=vegetation_type_pollen,
            bloom_coverage_km2=bloom_area,
            bloom_intensity=bloom_intensity_pollen,
            day_of_year=day_of_year_pollen
        )
        
        st.success("✅ Pollen Forecast Generated!")
        
        # Risk alert
        risk_color = {
            'Low': 'green',
            'Moderate': 'orange',
            'High': 'red',
            'Very High': 'darkred'
        }
        
        st.markdown(f"""
        <div style='padding: 20px; background-color: {risk_color.get(pollen_result['risk_level'], 'gray')}; 
        color: white; border-radius: 10px; text-align: center;'>
            <h2>Allergy Risk: {pollen_result['risk_level']}</h2>
            <h3>Pollen Level: {pollen_result['pollen_level']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning(f"⚠️ **Advisory:** {pollen_result['advisory']}")
        
        # Pollen metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Score",
                f"{pollen_result['allergy_risk_score']:.1f}/100"
            )
        
        with col2:
            st.metric(
                "Affected Area",
                f"{pollen_result['affected_area_km2']:.1f} km²"
            )
        
        with col3:
            st.metric(
                "Peak Time",
                pollen_result['peak_pollen_times']['daily_peak']
            )
        
        # Peak pollen info
        st.subheader("⏰ Peak Pollen Information")
        
        peak_info = pollen_result['peak_pollen_times']
        
        st.markdown(f"""
        - **Season:** {peak_info['season']}
        - **Daily Peak:** {peak_info['daily_peak']}
        - **Weather Impact:** {peak_info['weather_dependency']}
        """)
        
        # Recommendations
        st.subheader("💡 Allergy Management Recommendations")
        
        for rec in pollen_result['recommendations']:
            st.markdown(f"✓ {rec}")

with tab4:
    st.header("Ecosystem Services Valuation")
    
    st.markdown("""
    Assess the economic value of ecosystem services provided by bloom events.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        bloom_area_eco = st.number_input(
            "Bloom Area (km²)",
            min_value=0.1,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            key='eco_area'
        )
        
        vegetation_type_eco = st.selectbox(
            "Vegetation Type",
            options=['forest', 'wetland', 'grassland', 'agricultural', 'wildflowers'],
            key='eco_veg_type'
        )
    
    with col2:
        bloom_health_eco = st.slider(
            "Bloom Health (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            key='eco_health',
            help="Overall health and vigor of the bloom"
        )
    
    if st.button("💰 Calculate Ecosystem Value"):
        # Assess ecosystem services
        eco_result = analyzer.assess_ecosystem_services(
            bloom_area_km2=bloom_area_eco,
            vegetation_type=vegetation_type_eco,
            bloom_health=bloom_health_eco
        )
        
        st.success("✅ Ecosystem Services Assessed!")
        
        # Total value
        total_value = eco_result['total_annual_value_usd']
        
        st.markdown(f"""
        <div style='padding: 20px; background-color: #2ecc71; color: white; 
        border-radius: 10px; text-align: center;'>
            <h2>Total Annual Value</h2>
            <h1>${total_value:,.0f} USD</h1>
            <p>Per km²: ${eco_result['per_km2_value_usd']:,.0f} USD</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key services
        st.subheader("🌿 Key Ecosystem Services")
        
        cols = st.columns(len(eco_result['key_services']))
        for col, service in zip(cols, eco_result['key_services']):
            with col:
                st.info(f"**{service}**")
        
        # Services breakdown
        st.subheader("📊 Services Valuation Breakdown")
        
        services_data = []
        for service_name, service_data in eco_result['services_breakdown'].items():
            services_data.append({
                'Service': service_name.replace('_', ' ').title(),
                'Annual Value (USD)': service_data['annual_value_usd'],
                'Per Hectare (USD)': service_data['per_hectare_usd']
            })
        
        services_df = pd.DataFrame(services_data)
        services_df = services_df.sort_values('Annual Value (USD)', ascending=False)
        
        # Bar chart
        fig_services = go.Figure(data=[go.Bar(
            x=services_df['Service'],
            y=services_df['Annual Value (USD)'],
            marker_color='#3498db',
            text=services_df['Annual Value (USD)'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        )])
        
        fig_services.update_layout(
            title="Ecosystem Services Value by Category",
            xaxis_title="Service",
            yaxis_title="Annual Value (USD)",
            height=500
        )
        
        st.plotly_chart(fig_services, use_container_width=True)
        
        # Detailed table
        st.dataframe(
            services_df,
            use_container_width=True,
            column_config={
                'Service': st.column_config.TextColumn('Service', width='medium'),
                'Annual Value (USD)': st.column_config.NumberColumn(
                    'Annual Value (USD)',
                    format='$%,.0f'
                ),
                'Per Hectare (USD)': st.column_config.NumberColumn(
                    'Per Hectare (USD)',
                    format='$%,.2f'
                )
            },
            hide_index=True
        )

with tab5:
    st.header("Comprehensive Impact Report")
    
    st.markdown("""
    Generate a complete ecological impact report combining all analyses.
    """)
    
    # Report parameters
    st.subheader("📋 Report Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        region_name = st.text_input(
            "Region Name",
            value="California Central Valley"
        )
        
        report_coverage = st.slider(
            "Bloom Coverage (%)",
            0.0, 100.0, 70.0, 5.0,
            key='report_coverage'
        )
        
        report_intensity = st.slider(
            "Bloom Intensity (NDVI)",
            0.0, 1.0, 0.68, 0.05,
            key='report_intensity'
        )
    
    with col2:
        report_date = st.date_input(
            "Report Date",
            value=datetime.now()
        )
        
        report_area = st.number_input(
            "Bloom Area (km²)",
            1.0, 10000.0, 250.0, 10.0,
            key='report_area'
        )
        
        report_veg_type = st.selectbox(
            "Vegetation Type",
            ['wildflowers', 'orchard', 'agricultural', 'grassland', 'forest'],
            key='report_veg'
        )
    
    if st.button("📄 Generate Impact Report", type="primary"):
        # Prepare bloom data
        bloom_data = {
            'coverage_percent': report_coverage,
            'intensity': report_intensity,
            'area_km2': report_area,
            'vegetation_type': report_veg_type
        }
        
        # Generate report
        report_datetime = datetime.combine(report_date, datetime.min.time())
        
        with st.spinner("Generating comprehensive impact report..."):
            report = analyzer.generate_impact_report(
                bloom_data=bloom_data,
                region_name=region_name,
                date=report_datetime
            )
        
        st.success("✅ Impact Report Generated!")
        
        # Display report
        st.markdown("---")
        st.header(f"🌸 Ecological Impact Report: {region_name}")
        st.markdown(f"**Report Date:** {report['report_date'][:10]}")
        
        # Bloom summary
        st.subheader("📊 Bloom Summary")
        summary = report['bloom_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Coverage", f"{summary['coverage_percent']:.0f}%")
        
        with col2:
            st.metric("Intensity", f"{summary['intensity']:.2f}")
        
        with col3:
            st.metric("Area", f"{summary['area_km2']:.0f} km²")
        
        with col4:
            st.metric("Type", summary['vegetation_type'].title())
        
        # Overall assessment
        st.info(f"**Overall Assessment:** {report['overall_assessment']}")
        
        # Key findings
        st.subheader("🔍 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🐝 Pollinator Activity**")
            poll = report['pollinator_activity']
            st.write(f"- Activity Score: {poll['activity_score']:.1f}/100")
            st.write(f"- Level: {poll['activity_level']}")
            
            st.markdown("**🌾 Crop Yield Forecast**")
            crop = report['crop_yield_forecast']
            st.write(f"- Yield Index: {crop['yield_index']:.1f}/100")
            st.write(f"- Outlook: {crop['outlook']}")
        
        with col2:
            st.markdown("**🌼 Pollen Forecast**")
            pollen = report['pollen_forecast']
            st.write(f"- Risk Level: {pollen['risk_level']}")
            st.write(f"- Advisory: {pollen['advisory'][:50]}...")
            
            st.markdown("**💰 Ecosystem Services**")
            eco = report['ecosystem_services']
            st.write(f"- Total Value: ${eco['total_annual_value_usd']:,.0f}")
            st.write(f"- Per km²: ${eco['per_km2_value_usd']:,.0f}")
        
        # Download report
        st.markdown("---")
        
        # Create downloadable report
        report_text = f"""
ECOLOGICAL IMPACT REPORT
========================

Region: {region_name}
Date: {report['report_date'][:10]}

BLOOM SUMMARY
-------------
Coverage: {summary['coverage_percent']:.1f}%
Intensity: {summary['intensity']:.2f}
Area: {summary['area_km2']:.1f} km²
Vegetation Type: {summary['vegetation_type']}

OVERALL ASSESSMENT
------------------
{report['overall_assessment']}

POLLINATOR ACTIVITY
-------------------
Activity Score: {poll['activity_score']:.1f}/100
Activity Level: {poll['activity_level']}
Description: {poll['description']}

CROP YIELD FORECAST
-------------------
Yield Index: {crop['yield_index']:.1f}/100
Outlook: {crop['outlook']}
Expected Yield: {crop['expected_yield_pct']}

POLLEN FORECAST
---------------
Risk Level: {pollen['risk_level']}
Allergy Risk Score: {pollen['allergy_risk_score']:.1f}/100
Advisory: {pollen['advisory']}

ECOSYSTEM SERVICES
------------------
Total Annual Value: ${eco['total_annual_value_usd']:,.0f} USD
Value per km²: ${eco['per_km2_value_usd']:,.0f} USD
Key Services: {', '.join(eco['key_services'])}

---
Generated by BloomSphere - NASA Space Apps Challenge
"""
        
        st.download_button(
            label="📥 Download Full Report (TXT)",
            data=report_text,
            file_name=f"bloom_impact_report_{region_name.replace(' ', '_')}_{report_date}.txt",
            mime="text/plain"
        )

# Sidebar
with st.sidebar:
    st.header("ℹ️ About Ecological Impacts")
    
    st.markdown("""
    ### Impact Categories
    
    **Pollinator Activity**
    - Bee and butterfly populations
    - Foraging patterns
    - Habitat quality assessment
    
    **Agricultural Impacts**
    - Crop yield forecasts
    - Pollination services
    - Economic implications
    
    **Public Health**
    - Pollen forecasts
    - Allergy risk levels
    - Health advisories
    
    **Ecosystem Services**
    - Carbon sequestration
    - Water regulation
    - Biodiversity support
    - Economic valuation
    
    ### Applications
    - Agriculture planning
    - Apiary management
    - Public health advisories
    - Conservation prioritization
    - Policy development
    """)
