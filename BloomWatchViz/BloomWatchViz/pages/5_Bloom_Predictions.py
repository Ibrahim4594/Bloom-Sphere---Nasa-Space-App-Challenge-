"""
Bloom Predictions Page for BloomSphere
ML-based predictions of future bloom events
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.bloom_predictor import BloomPredictor
from utils.ui_components import (
    page_header, section_header, info_panel,
    create_professional_chart_layout, COLORS
)

st.set_page_config(
    page_title="Bloom Predictions - BloomSphere",
    page_icon="🔮",
    layout="wide"
)

page_header(
    "Bloom Event Predictions",
    "Predict future bloom events using machine learning and historical vegetation patterns. Our models analyze NDVI time-series data to forecast bloom timing, intensity, and duration.",
    icon="🔮"
)

# Initialize predictor
@st.cache_resource
def get_predictor():
    """Get or create bloom predictor"""
    return BloomPredictor()

predictor = get_predictor()

# Tabs for different prediction features
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historical Analysis",
    "🔮 Next Bloom Prediction",
    "📈 Seasonal Forecast",
    "📅 Bloom Calendar"
])

with tab1:
    section_header("Historical Bloom Pattern Analysis")
    
    # Generate sample historical data for demonstration
    @st.cache_data
    def generate_sample_historical_data(years=3):
        """Generate sample NDVI time series with seasonal blooms"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='8D')
        
        ndvi_values = []
        for date in dates:
            # Day of year for seasonality
            day_of_year = date.timetuple().tm_yday
            
            # Base vegetation level
            base_ndvi = 0.35
            
            # Seasonal bloom (peaks around day 120 - late April/early May)
            peak_day = 120
            distance_from_peak = min(abs(day_of_year - peak_day), 365 - abs(day_of_year - peak_day))
            seasonal_bloom = 0.4 * np.exp(-(distance_from_peak ** 2) / (2 * 30 ** 2))
            
            # Add some inter-annual variability
            year_factor = 1.0 + 0.1 * np.sin(date.year * 0.5)
            
            # Add noise
            noise = np.random.normal(0, 0.05)
            
            ndvi = base_ndvi + (seasonal_bloom * year_factor) + noise
            ndvi = np.clip(ndvi, 0, 1)
            
            ndvi_values.append(ndvi)
        
        return pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values
        })
    
    # Get sample data
    historical_df = generate_sample_historical_data(years=3)
    
    st.markdown(f"<h3 style='color: {COLORS['text_dark']}; margin-top: 1.5rem;'>📈 NDVI Time Series</h3>", unsafe_allow_html=True)
    
    # Plot historical NDVI
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['ndvi'],
        mode='lines+markers',
        name='NDVI',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=4)
    ))
    
    # Add bloom threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Bloom Threshold",
        annotation_position="right"
    )
    
    layout_config = create_professional_chart_layout()
    fig.update_layout(
        title="Historical Vegetation Index (NDVI) - 3 Years",
        xaxis_title="Date",
        yaxis_title="NDVI",
        hovermode='x unified',
        height=400,
        **layout_config
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Detect cycles
    st.markdown(f"<h3 style='color: {COLORS['text_dark']}; margin-top: 1.5rem;'>🔄 Detected Bloom Cycles</h3>", unsafe_allow_html=True)
    
    with st.spinner("Analyzing bloom patterns..."):
        cycles = predictor.detect_bloom_cycles(
            historical_df['date'].tolist(),
            historical_df['ndvi'].tolist(),
            min_cycle_days=90
        )
    
    if cycles:
        st.success(f"✅ Detected {len(cycles)} bloom cycles")
        
        # Display cycles table
        cycles_df = pd.DataFrame(cycles)
        
        st.dataframe(
            cycles_df[[
                'cycle_number', 'peak_date', 'start_date', 'end_date',
                'peak_ndvi', 'duration_days', 'month'
            ]],
            use_container_width=True,
            column_config={
                'cycle_number': st.column_config.NumberColumn('Cycle #', width='small'),
                'peak_date': st.column_config.DateColumn('Peak Date', width='medium'),
                'start_date': st.column_config.DateColumn('Start Date', width='medium'),
                'end_date': st.column_config.DateColumn('End Date', width='medium'),
                'peak_ndvi': st.column_config.NumberColumn('Peak NDVI', format='%.3f'),
                'duration_days': st.column_config.NumberColumn('Duration (days)'),
                'month': st.column_config.NumberColumn('Month')
            },
            hide_index=True
        )
        
        # Seasonal pattern analysis
        st.subheader("🌸 Seasonal Pattern Analysis")
        
        pattern = predictor.calculate_seasonal_pattern(cycles)
        
        if pattern['has_pattern']:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Typical Season",
                    pattern.get('typical_season', 'Unknown')
                )
            
            with col2:
                st.metric(
                    "Avg Peak Day",
                    f"Day {pattern['avg_peak_day']}",
                    delta=f"± {pattern['std_peak_day']:.0f} days"
                )
            
            with col3:
                st.metric(
                    "Avg Peak NDVI",
                    f"{pattern['avg_peak_ndvi']:.3f}",
                    delta=f"± {pattern['std_peak_ndvi']:.3f}"
                )
            
            with col4:
                consistency_pct = pattern['consistency_score'] * 100
                st.metric(
                    "Pattern Consistency",
                    f"{consistency_pct:.0f}%"
                )
            
            # Bloom duration distribution
            st.subheader("⏱️ Bloom Duration Distribution")
            
            durations = [c['duration_days'] for c in cycles]
            
            fig_dur = go.Figure()
            fig_dur.add_trace(go.Histogram(
                x=durations,
                nbinsx=10,
                marker_color='#3498db',
                name='Duration'
            ))
            
            fig_dur.update_layout(
                title="Distribution of Bloom Durations",
                xaxis_title="Duration (days)",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig_dur, use_container_width=True)
    
    else:
        st.warning("⚠️ No bloom cycles detected in historical data")
    
    # Trend analysis
    st.subheader("📊 Long-term Trend Analysis")
    
    trend = predictor.analyze_trend(
        historical_df['date'].tolist(),
        historical_df['ndvi'].tolist()
    )
    
    if trend['has_trend']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Trend Direction", trend['direction'])
        
        with col2:
            change_pct = abs(trend['yearly_change']) * 100
            st.metric("Yearly Change", f"{change_pct:.2f}%/year")
        
        with col3:
            confidence_pct = trend['r_squared'] * 100
            st.metric("Confidence", f"{confidence_pct:.0f}%")
        
        st.info(f"**Interpretation:** {trend['interpretation']}")

with tab2:
    st.header("🔮 Next Bloom Prediction")
    
    # Use historical data from tab1
    cycles = predictor.detect_bloom_cycles(
        historical_df['date'].tolist(),
        historical_df['ndvi'].tolist(),
        min_cycle_days=90
    )
    
    if cycles:
        prediction = predictor.predict_next_bloom(cycles)
        
        if prediction['predicted']:
            st.success("✅ Next bloom event predicted!")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Predicted Date",
                    prediction['predicted_date'].strftime('%B %d, %Y')
                )
            
            with col2:
                st.metric(
                    "Days Until Bloom",
                    f"{prediction['days_until_bloom']} days"
                )
            
            with col3:
                confidence_pct = prediction['confidence'] * 100
                st.metric(
                    "Confidence",
                    f"{confidence_pct:.0f}%",
                    delta=prediction['confidence_level']
                )
            
            with col4:
                st.metric(
                    "Expected Peak NDVI",
                    f"{prediction['predicted_peak_ndvi']:.3f}"
                )
            
            # Detailed prediction
            st.subheader("📋 Prediction Details")
            
            details_df = pd.DataFrame([{
                'Metric': 'Predicted Season',
                'Value': prediction['typical_season']
            }, {
                'Metric': 'Expected Duration',
                'Value': f"{prediction['predicted_duration_days']} days"
            }, {
                'Metric': 'Uncertainty Range',
                'Value': f"± {prediction['uncertainty_days']} days"
            }, {
                'Metric': 'Based on Historical Cycles',
                'Value': f"{prediction['based_on_cycles']} cycles"
            }])
            
            st.table(details_df)
            
            # Visual timeline
            st.subheader("📅 Bloom Timeline")
            
            today = datetime.now()
            bloom_date = prediction['predicted_date']
            
            timeline_data = []
            for i in range(-30, prediction['days_until_bloom'] + 30, 5):
                date = today + timedelta(days=i)
                
                # Calculate expected NDVI
                days_from_bloom = (date - bloom_date).days
                bloom_curve = np.exp(-(days_from_bloom ** 2) / (2 * (prediction['predicted_duration_days']/3) ** 2))
                expected_ndvi = 0.35 + (prediction['predicted_peak_ndvi'] - 0.35) * bloom_curve
                
                timeline_data.append({
                    'date': date,
                    'expected_ndvi': expected_ndvi
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig_timeline = go.Figure()
            
            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['expected_ndvi'],
                mode='lines',
                name='Expected NDVI',
                line=dict(color='#2ecc71', width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))
            
            # Mark today
            fig_timeline.add_vline(
                x=today.timestamp() * 1000,  # Convert to milliseconds
                line_dash="dash",
                line_color="blue",
                annotation_text="Today",
                annotation_position="top"
            )
            
            # Mark predicted bloom
            fig_timeline.add_vline(
                x=bloom_date.timestamp() * 1000,  # Convert to milliseconds
                line_dash="dash",
                line_color="red",
                annotation_text="Predicted Peak",
                annotation_position="top"
            )
            
            fig_timeline.update_layout(
                title="Predicted Bloom Curve",
                xaxis_title="Date",
                yaxis_title="Expected NDVI",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        else:
            st.warning(f"⚠️ {prediction['message']}")
    
    else:
        st.info("ℹ️ Analyze historical data in the first tab to generate predictions")

with tab3:
    st.header("📈 Seasonal Forecast (6 Months)")
    
    st.markdown("""
    Generate a 6-month forecast of vegetation indices based on historical patterns.
    """)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            forecast_df = predictor.generate_forecast(historical_df, forecast_days=180)
        
        if not forecast_df.empty:
            # Combine historical and forecast
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df['ndvi'],
                mode='lines',
                name='Historical NDVI',
                line=dict(color='#2ecc71', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast_ndvi'],
                mode='lines',
                name='Forecast NDVI',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Confidence Interval'
            ))
            
            fig_forecast.update_layout(
                title="6-Month Vegetation Index Forecast",
                xaxis_title="Date",
                yaxis_title="NDVI",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast statistics
            st.subheader("📊 Forecast Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = forecast_df['forecast_ndvi'].mean()
                st.metric("Average NDVI", f"{avg_forecast:.3f}")
            
            with col2:
                max_forecast = forecast_df['forecast_ndvi'].max()
                max_date = forecast_df.loc[forecast_df['forecast_ndvi'].idxmax(), 'date']
                st.metric(
                    "Peak NDVI",
                    f"{max_forecast:.3f}",
                    delta=max_date.strftime('%b %d')
                )
            
            with col3:
                bloom_days = len(forecast_df[forecast_df['forecast_ndvi'] > 0.5])
                st.metric("Bloom Days", f"{bloom_days} days")
            
            # Download forecast
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name=f"bloom_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        else:
            st.error("Unable to generate forecast. Please check historical data.")

with tab4:
    st.header("📅 Bloom Probability Calendar")
    
    st.markdown("""
    View the probability of bloom events for specific dates based on historical patterns.
    """)
    
    # Date selection
    target_date = st.date_input(
        "Select Date to Check Bloom Probability",
        value=datetime.now() + timedelta(days=60),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=365)
    )
    
    if st.button("Calculate Bloom Probability"):
        target_datetime = datetime.combine(target_date, datetime.min.time())
        
        prob_result = predictor.calculate_bloom_probability(
            historical_df,
            target_datetime
        )
        
        # Display probability
        prob_pct = prob_result['probability'] * 100
        
        st.subheader(f"Bloom Probability for {target_date.strftime('%B %d, %Y')}")
        
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bloom Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2ecc71" if prob_pct > 60 else "#f39c12" if prob_pct > 30 else "#e74c3c"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "gray"},
                    {'range': [60, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Level", prob_result['confidence'])
        
        with col2:
            st.metric(
                "Nearest Historical Bloom",
                f"{prob_result['days_to_nearest_bloom']} days away"
            )
        
        with col3:
            st.metric(
                "Pattern Consistency",
                f"{prob_result['pattern_consistency']*100:.0f}%"
            )
        
        # Interpretation
        if prob_pct >= 70:
            st.success("🌸 **High probability of bloom!** This date aligns well with historical bloom patterns.")
        elif prob_pct >= 40:
            st.warning("🌼 **Moderate probability of bloom.** There's a reasonable chance of bloom activity around this date.")
        else:
            st.info("🍂 **Low probability of bloom.** This date is outside typical bloom periods.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About Predictions")
    
    st.markdown("""
    ### Prediction Methods
    
    **Cycle Detection**
    - Identifies recurring bloom patterns
    - Uses peak detection algorithms
    - Requires 10+ historical observations
    
    **Trend Analysis**
    - Linear regression on NDVI time series
    - Detects greening/browning trends
    - Provides long-term projections
    
    **Seasonal Forecasting**
    - Pattern-based predictions
    - Accounts for inter-annual variability
    - 6-month rolling forecasts
    
    ### Data Requirements
    - Minimum 30 observations for forecasts
    - 10+ observations for cycle detection
    - Regular temporal spacing (e.g., 8-16 days)
    
    ### Accuracy Notes
    - Predictions improve with more data
    - Weather events may affect accuracy
    - Use predictions as guidance, not certainty
    """)
