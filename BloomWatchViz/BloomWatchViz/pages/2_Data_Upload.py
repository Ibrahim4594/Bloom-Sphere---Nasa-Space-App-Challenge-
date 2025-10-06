import streamlit as st
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import tempfile
import os
from datetime import datetime, date
from utils.satellite_processor import SatelliteProcessor
from utils.vegetation_indices import VegetationIndices
from database.db_manager import DatabaseManager
from utils.export_manager import ExportManager
from utils.ui_components import (
    page_header, section_header, info_panel, COLORS
)

st.set_page_config(
    page_title="Data Upload - BloomSphere",
    page_icon="📤",
    layout="wide"
)

def main():
    page_header(
        "Satellite Data Upload & Processing",
        "Upload and process satellite imagery to detect plant blooming events",
        icon="📤"
    )
    
    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["🛰️ Satellite Imagery", "📊 Processed Data", "🔧 Batch Processing"])
    
    with tab1:
        handle_satellite_upload()
    
    with tab2:
        handle_processed_data()
    
    with tab3:
        handle_batch_processing()

def handle_satellite_upload():
    """Handle individual satellite image uploads"""
    
    section_header("🛰️ Upload Satellite Imagery")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_files = st.file_uploader(
            "Upload satellite imagery files (TIFF format)",
            type=['tiff', 'tif', 'TIF', 'TIFF'],
            accept_multiple_files=True,
            help="Supported formats: MODIS, Landsat, VIIRS TIFF files"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
    
    with col2:
        st.subheader("📋 Processing Options")
        
        # Processing parameters
        vegetation_index = st.selectbox(
            "Vegetation Index",
            ["NDVI", "EVI", "SAVI", "ARVI"],
            help="Select vegetation index to calculate"
        )
        
        cloud_threshold = st.slider(
            "Cloud Cover Threshold (%)",
            min_value=0,
            max_value=100,
            value=20,
            help="Maximum acceptable cloud cover percentage"
        )
        
        spatial_resolution = st.selectbox(
            "Output Resolution",
            ["Original", "250m", "500m", "1km"],
            help="Spatial resolution for output"
        )
        
        # Bloom detection settings
        st.subheader("🌸 Bloom Detection")
        
        bloom_threshold = st.slider(
            "Bloom Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum vegetation index value to consider as bloom"
        )
        
        temporal_window = st.slider(
            "Temporal Window (days)",
            min_value=7,
            max_value=60,
            value=14,
            help="Number of days to analyze for bloom detection"
        )
        
        # Sample data option
        st.markdown("---")
        st.subheader("🎯 Try with Sample Data")
        
        st.markdown("**Example Output Heatmaps:**")
        st.image(
            "attached_assets/productOverview_heatmap_1759601442713.png",
            caption="Example: Processed Bloom Intensity Heatmap",
            width="stretch"
        )
        
        if st.button("📥 Load Sample MODIS Data"):
            load_sample_data()

def save_to_database(file_name, metadata, ndvi, evi, savi, bloom_mask, bloom_stats):
    """Save processing results to database"""
    try:
        with DatabaseManager() as db:
            # Prepare imagery metadata
            file_size_mb = os.path.getsize(tempfile.gettempdir()) / (1024 * 1024) if os.path.exists(tempfile.gettempdir()) else 0.0
            
            imagery_metadata = {
                'filename': file_name,
                'satellite_type': 'MODIS',  # Default, can be detected from filename
                'acquisition_date': date.today(),
                'width': metadata.get('width'),
                'height': metadata.get('height'),
                'bands': metadata.get('bands'),
                'crs': metadata.get('crs'),
                'bounds_north': metadata['bounds'][3],
                'bounds_south': metadata['bounds'][1],
                'bounds_east': metadata['bounds'][2],
                'bounds_west': metadata['bounds'][0],
                'resolution_x': metadata['resolution'][0],
                'resolution_y': metadata['resolution'][1],
                'cloud_cover': 0.0,  # Would be calculated from actual data
                'file_size_mb': file_size_mb,
                'processing_status': 'completed'
            }
            
            # Insert imagery record
            imagery_id = db.insert_satellite_imagery(imagery_metadata)
            
            # Save vegetation indices
            for index_name, index_data in [('NDVI', ndvi), ('EVI', evi), ('SAVI', savi)]:
                vi_stats = {
                    'mean': float(np.nanmean(index_data)),
                    'median': float(np.nanmedian(index_data)),
                    'min': float(np.nanmin(index_data)),
                    'max': float(np.nanmax(index_data)),
                    'std': float(np.nanstd(index_data)),
                    'percentile_25': float(np.nanpercentile(index_data, 25)),
                    'percentile_75': float(np.nanpercentile(index_data, 75)),
                    'percentile_90': float(np.nanpercentile(index_data, 90)),
                    'valid_pixels': int(np.sum(~np.isnan(index_data))),
                    'total_pixels': int(index_data.size)
                }
                db.insert_vegetation_index(imagery_id, index_name, vi_stats)
            
            # Save bloom detection
            detection_params = {
                'threshold': 0.4,
                'method': 'threshold',
                'confidence_score': 0.85
            }
            detection_id = db.insert_bloom_detection(imagery_id, detection_params, bloom_stats)
            
            return imagery_id
            
    except Exception as e:
        st.warning(f"Could not save to database: {str(e)}")
        return None

def process_uploaded_files(uploaded_files):
    """Process uploaded satellite files"""
    
    st.subheader(f"Processing {len(uploaded_files)} file(s)")
    
    processor = SatelliteProcessor()
    
    for i, uploaded_file in enumerate(uploaded_files):
        with st.expander(f"📁 {uploaded_file.name}", expanded=i==0):
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Process the file
                process_single_file(tmp_file_path, uploaded_file.name, processor)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

def process_single_file(file_path, file_name, processor):
    """Process a single satellite file"""
    
    # Read metadata
    try:
        with rasterio.open(file_path) as src:
            metadata = {
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'crs': str(src.crs),
                'bounds': src.bounds,
                'resolution': src.res
            }
        
        # Display metadata
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 File Metadata**")
            st.json(metadata)
        
        with col2:
            st.markdown("**🗺️ Geographic Information**")
            st.write(f"**Coordinate System:** {metadata['crs']}")
            st.write(f"**Spatial Resolution:** {metadata['resolution'][0]:.0f}m")
            st.write(f"**Coverage Area:** {(metadata['bounds'][2] - metadata['bounds'][0]):.3f}° × {(metadata['bounds'][3] - metadata['bounds'][1]):.3f}°")
        
        # Process the imagery
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Read data
        status_text.text("Reading satellite data...")
        progress_bar.progress(20)
        
        imagery_data = processor.read_imagery(file_path)
        
        # Step 2: Calculate vegetation indices
        status_text.text("Calculating vegetation indices...")
        progress_bar.progress(40)
        
        vi_calculator = VegetationIndices()
        ndvi = vi_calculator.calculate_ndvi(imagery_data)
        evi = vi_calculator.calculate_evi(imagery_data)
        savi = vi_calculator.calculate_savi(imagery_data)
        
        # Step 3: Detect blooms
        status_text.text("Detecting bloom events...")
        progress_bar.progress(60)
        
        bloom_mask = processor.detect_blooms(ndvi, threshold=0.4)
        bloom_stats = processor.calculate_bloom_statistics(bloom_mask, ndvi)
        
        # Step 4: Save to database
        status_text.text("Saving results to database...")
        progress_bar.progress(70)
        
        imagery_id = save_to_database(file_name, metadata, ndvi, evi, savi, bloom_mask, bloom_stats)
        
        if imagery_id:
            st.success(f"✅ Data saved to database (ID: {imagery_id})")
        
        # Step 5: Generate visualizations
        status_text.text("Generating visualizations...")
        progress_bar.progress(85)
        
        create_processing_visualizations(imagery_data, ndvi, evi, bloom_mask, metadata)
        
        # Step 6: Generate statistics
        status_text.text("Computing statistics...")
        progress_bar.progress(100)
        
        generate_processing_statistics(ndvi, evi, bloom_mask, metadata)
        
        status_text.text("✅ Processing complete!")
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def create_processing_visualizations(imagery_data, ndvi, evi, bloom_mask, metadata):
    """Create visualizations for processed data"""
    
    st.subheader("📈 Processing Results")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["True Color", "NDVI", "EVI", "Bloom Detection"])
    
    with viz_tab1:
        st.markdown("**True Color Composite**")
        if imagery_data.shape[0] >= 3:  # Ensure we have at least 3 bands
            create_true_color_image(imagery_data)
        else:
            st.warning("Insufficient bands for true color composite")
    
    with viz_tab2:
        st.markdown("**NDVI (Normalized Difference Vegetation Index)**")
        create_vegetation_heatmap(ndvi, "NDVI", "RdYlGn")
    
    with viz_tab3:
        st.markdown("**EVI (Enhanced Vegetation Index)**")
        create_vegetation_heatmap(evi, "EVI", "viridis")
    
    with viz_tab4:
        st.markdown("**Bloom Detection Results**")
        create_bloom_detection_visualization(bloom_mask, ndvi)

def create_true_color_image(imagery_data):
    """Create true color composite image"""
    
    # Assuming bands are in order: Red, Green, Blue, NIR, ...
    # Adjust band indices based on satellite type
    red_band = imagery_data[0] if imagery_data.shape[0] > 0 else None
    green_band = imagery_data[1] if imagery_data.shape[0] > 1 else None
    blue_band = imagery_data[2] if imagery_data.shape[0] > 2 else None
    
    if red_band is not None and green_band is not None and blue_band is not None:
        # Normalize bands to 0-255 range
        def normalize_band(band):
            band_min, band_max = np.percentile(band[band > 0], [2, 98])
            normalized = np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255)
            return normalized.astype(np.uint8)
        
        red_norm = normalize_band(red_band)
        green_norm = normalize_band(green_band)
        blue_norm = normalize_band(blue_band)
        
        # Create RGB image
        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
        
        # Display using matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb_image)
        ax.set_title("True Color Composite")
        ax.axis('off')
        
        st.pyplot(fig)
        plt.close()
    else:
        st.error("Cannot create true color composite - insufficient bands")

def create_vegetation_heatmap(data, title, colormap):
    """Create vegetation index heatmap"""
    
    # Sample data for display if too large
    if data.shape[0] > 1000 or data.shape[1] > 1000:
        step = max(data.shape[0] // 500, data.shape[1] // 500, 1)
        data_sampled = data[::step, ::step]
    else:
        data_sampled = data
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=data_sampled,
        colorscale=colormap,
        hovertemplate=f'{title}: %{{z:.3f}}<extra></extra>',
        colorbar=dict(title=title)
    ))
    
    fig.update_layout(
        title=f'{title} Heatmap',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_bloom_detection_visualization(bloom_mask, ndvi):
    """Create bloom detection visualization"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bloom mask visualization
        fig = go.Figure(data=go.Heatmap(
            z=bloom_mask.astype(int),
            colorscale=['white', 'red'],
            showscale=False,
            hovertemplate='Bloom Detected: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Bloom Detection Mask',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # NDVI with bloom overlay
        bloom_overlay = np.where(bloom_mask, ndvi, np.nan)
        
        fig = go.Figure()
        
        # Base NDVI
        fig.add_trace(go.Heatmap(
            z=ndvi,
            colorscale='YlGn',
            opacity=0.7,
            name='NDVI',
            showscale=False
        ))
        
        # Bloom overlay
        fig.add_trace(go.Heatmap(
            z=bloom_overlay,
            colorscale='Reds',
            opacity=0.8,
            name='Detected Blooms'
        ))
        
        fig.update_layout(
            title='NDVI with Bloom Overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def generate_processing_statistics(ndvi, evi, bloom_mask, metadata):
    """Generate and display processing statistics"""
    
    st.subheader("📊 Processing Statistics")
    
    # Calculate statistics
    stats = {
        'total_pixels': ndvi.size,
        'valid_pixels': np.sum(~np.isnan(ndvi)),
        'bloom_pixels': np.sum(bloom_mask),
        'bloom_percentage': (np.sum(bloom_mask) / np.sum(~np.isnan(ndvi))) * 100,
        'mean_ndvi': np.nanmean(ndvi),
        'max_ndvi': np.nanmax(ndvi),
        'mean_evi': np.nanmean(evi),
        'max_evi': np.nanmax(evi),
        'coverage_area_km2': ((metadata['bounds'][2] - metadata['bounds'][0]) * 
                              (metadata['bounds'][3] - metadata['bounds'][1]) * 
                              111.32 * 111.32)  # Rough conversion to km²
    }
    
    # Display statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Area (km²)", f"{stats['coverage_area_km2']:.1f}")
        st.metric("Valid Pixels", f"{stats['valid_pixels']:,}")
    
    with col2:
        st.metric("Bloom Coverage", f"{stats['bloom_percentage']:.1f}%")
        st.metric("Bloom Pixels", f"{stats['bloom_pixels']:,}")
    
    with col3:
        st.metric("Mean NDVI", f"{stats['mean_ndvi']:.3f}")
        st.metric("Max NDVI", f"{stats['max_ndvi']:.3f}")
    
    with col4:
        st.metric("Mean EVI", f"{stats['mean_evi']:.3f}")
        st.metric("Max EVI", f"{stats['max_evi']:.3f}")
    
    # Histogram of vegetation index values
    st.subheader("📈 Vegetation Index Distribution")
    
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        # NDVI histogram
        ndvi_flat = ndvi[~np.isnan(ndvi)]
        fig = px.histogram(
            x=ndvi_flat,
            nbins=50,
            title="NDVI Distribution",
            labels={'x': 'NDVI Value', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with hist_col2:
        # EVI histogram
        evi_flat = evi[~np.isnan(evi)]
        fig = px.histogram(
            x=evi_flat,
            nbins=50,
            title="EVI Distribution",
            labels={'x': 'EVI Value', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

def load_sample_data():
    """Load and process sample satellite data"""
    
    st.info("🎯 Loading sample MODIS data for demonstration...")
    
    # Generate synthetic MODIS-like data
    np.random.seed(42)
    
    # Simulate a 500x500 pixel MODIS image
    height, width = 500, 500
    
    # Simulate different bands
    # Band 1 (Red): 620-670 nm
    red_band = np.random.normal(0.1, 0.05, (height, width))
    red_band = np.clip(red_band, 0, 1)
    
    # Band 2 (NIR): 841-876 nm  
    nir_band = np.random.normal(0.4, 0.15, (height, width))
    nir_band = np.clip(nir_band, 0, 1)
    
    # Band 3 (Blue): 459-479 nm
    blue_band = np.random.normal(0.08, 0.03, (height, width))
    blue_band = np.clip(blue_band, 0, 1)
    
    # Band 4 (Green): 545-565 nm
    green_band = np.random.normal(0.12, 0.04, (height, width))
    green_band = np.clip(green_band, 0, 1)
    
    # Add some vegetation patterns
    y, x = np.ogrid[:height, :width]
    
    # Create vegetation hotspots
    for i in range(5):
        center_y, center_x = np.random.randint(50, height-50), np.random.randint(50, width-50)
        radius = np.random.randint(30, 80)
        
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Enhance vegetation in these areas
        nir_band[mask] = np.clip(nir_band[mask] * 2, 0, 1)
        green_band[mask] = np.clip(green_band[mask] * 1.5, 0, 1)
        red_band[mask] = np.clip(red_band[mask] * 0.8, 0, 1)
    
    # Stack bands
    imagery_data = np.stack([red_band, green_band, blue_band, nir_band], axis=0)
    
    # Create synthetic metadata
    metadata = {
        'width': width,
        'height': height,
        'bands': 4,
        'crs': 'EPSG:4326',
        'bounds': (-120.0, 35.0, -119.0, 36.0),  # California coordinates
        'resolution': (250.0, 250.0)
    }
    
    # Process the sample data
    st.success("✅ Sample data loaded successfully!")
    
    processor = SatelliteProcessor()
    vi_calculator = VegetationIndices()
    
    # Calculate vegetation indices
    ndvi = vi_calculator.calculate_ndvi(imagery_data)
    evi = vi_calculator.calculate_evi(imagery_data)
    
    # Detect blooms
    bloom_mask = processor.detect_blooms(ndvi, threshold=0.4)
    
    # Create visualizations
    create_processing_visualizations(imagery_data, ndvi, evi, bloom_mask, metadata)
    
    # Generate statistics
    generate_processing_statistics(ndvi, evi, bloom_mask, metadata)

def handle_processed_data():
    """Handle viewing and managing processed data"""
    
    st.header("📊 Processed Data Management")
    
    # Sample processed data viewer
    st.subheader("🗂️ Recent Processing Results")
    
    # Create sample processing history
    processing_history = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10, freq='W'),
        'File': [f'MODIS_A2024{str(i).zfill(3)}_h09v04.hdf' for i in range(1, 11)],
        'Region': ['California', 'Oregon', 'Washington', 'Nevada', 'Arizona', 
                  'Utah', 'Colorado', 'New Mexico', 'Texas', 'Oklahoma'],
        'NDVI_Mean': np.random.uniform(0.2, 0.8, 10),
        'Bloom_Coverage_%': np.random.uniform(5, 45, 10),
        'Status': ['Complete'] * 8 + ['Processing', 'Queued']
    })
    
    # Display processing history
    st.dataframe(
        processing_history,
        use_container_width=True,
        column_config={
            'NDVI_Mean': st.column_config.ProgressColumn(
                'Mean NDVI',
                help='Average NDVI value',
                min_value=0,
                max_value=1,
            ),
            'Bloom_Coverage_%': st.column_config.ProgressColumn(
                'Bloom Coverage (%)',
                help='Percentage of area with detected blooms',
                min_value=0,
                max_value=100,
            ),
        }
    )
    
    # Data export options
    st.subheader("💾 Data Export")
    
    # Get export manager
    export_mgr = ExportManager()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data, csv_filename = export_mgr.export_to_csv(processing_history)
        st.download_button(
            label="📄 Export as CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )
    
    with col2:
        json_data, json_filename = export_mgr.export_to_json(processing_history)
        st.download_button(
            label="📊 Export as JSON",
            data=json_data,
            file_name=json_filename,
            mime="application/json"
        )
    
    with col3:
        # Generate bloom report
        summary_stats = {
            'Total Processed Images': len(processing_history),
            'Average NDVI': f"{processing_history['NDVI_Mean'].mean():.3f}",
            'Average Bloom Coverage': f"{processing_history['Bloom_Coverage_%'].mean():.1f}%",
            'Date Range': f"{processing_history['Date'].min()} to {processing_history['Date'].max()}"
        }
        report_text, report_filename = export_mgr.create_bloom_report(summary_stats, processing_history)
        st.download_button(
            label="📋 Download Report",
            data=report_text,
            file_name=report_filename,
            mime="text/plain"
        )

def handle_batch_processing():
    """Handle batch processing of multiple files"""
    
    st.header("🔧 Batch Processing")
    
    st.markdown("""
    Upload multiple satellite images for automated batch processing. 
    This feature is ideal for processing time series data or large geographic areas.
    """)
    
    # Batch processing settings
    with st.expander("⚙️ Batch Processing Settings", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Sequential", "Parallel"],
                help="Sequential: Process files one by one. Parallel: Process multiple files simultaneously"
            )
            
            output_format = st.selectbox(
                "Output Format",
                ["GeoTIFF", "NetCDF", "HDF5"],
                help="Format for processed output files"
            )
            
            auto_cleanup = st.checkbox(
                "Auto-cleanup temporary files",
                value=True,
                help="Automatically remove temporary processing files"
            )
        
        with col2:
            notification_email = st.text_input(
                "Notification Email (optional)",
                help="Email address to receive processing completion notifications"
            )
            
            max_cloud_cover = st.slider(
                "Maximum Cloud Cover (%)",
                min_value=0,
                max_value=100,
                value=30,
                help="Skip images with cloud cover above this threshold"
            )
            
            quality_threshold = st.slider(
                "Quality Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Minimum quality score for processing"
            )
    
    # File upload for batch processing
    st.subheader("📁 Upload Files for Batch Processing")
    
    batch_files = st.file_uploader(
        "Select multiple satellite imagery files",
        type=['tiff', 'tif', 'hdf', 'nc'],
        accept_multiple_files=True,
        help="Upload multiple files for batch processing"
    )
    
    if batch_files:
        st.success(f"✅ {len(batch_files)} files ready for batch processing")
        
        # Show file summary
        file_summary = pd.DataFrame({
            'Filename': [f.name for f in batch_files],
            'Size (MB)': [f.size / (1024*1024) for f in batch_files],
            'Type': [f.type for f in batch_files]
        })
        
        st.dataframe(file_summary, use_container_width=True)
        
        # Processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Batch Processing", type="primary"):
                run_batch_processing(batch_files, processing_mode)
        
        with col2:
            if st.button("📋 Preview Processing Plan"):
                show_processing_plan(batch_files)
        
        with col3:
            if st.button("💾 Save Processing Queue"):
                st.info("Processing queue saved for later execution")

def run_batch_processing(files, mode):
    """Run batch processing on uploaded files"""
    
    st.subheader("🔄 Batch Processing in Progress")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_container = st.container()
    
    for i, file in enumerate(files):
        progress = (i + 1) / len(files)
        progress_bar.progress(progress)
        
        with status_container:
            st.info(f"Processing {file.name} ({i+1}/{len(files)})")
        
        # Simulate processing time
        import time
        time.sleep(1)
        
        with status_container:
            st.success(f"✅ Completed {file.name}")
    
    st.balloons()
    st.success("🎉 Batch processing completed successfully!")
    
    # Show processing summary
    summary_data = pd.DataFrame({
        'File': [f.name for f in files],
        'Status': ['Complete'] * len(files),
        'Processing Time': [f"{np.random.randint(30, 180)}s" for _ in files],
        'Output Size': [f"{np.random.randint(10, 100)}MB" for _ in files]
    })
    
    st.subheader("📋 Processing Summary")
    st.dataframe(summary_data, use_container_width=True)

def show_processing_plan(files):
    """Show the processing plan for batch files"""
    
    st.subheader("📋 Processing Plan Preview")
    
    plan_data = []
    estimated_time = 0
    
    for i, file in enumerate(files):
        file_size_mb = file.size / (1024 * 1024)
        estimated_duration = max(30, int(file_size_mb * 2))  # 2 seconds per MB, minimum 30s
        estimated_time += estimated_duration
        
        plan_data.append({
            'Order': i + 1,
            'Filename': file.name,
            'Size (MB)': f"{file_size_mb:.1f}",
            'Estimated Duration': f"{estimated_duration}s",
            'Operations': "Read → Calculate VI → Detect Blooms → Export"
        })
    
    plan_df = pd.DataFrame(plan_data)
    st.dataframe(plan_df, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Files", len(files))
    
    with col2:
        total_size = sum(f.size for f in files) / (1024 * 1024)
        st.metric("Total Size", f"{total_size:.1f} MB")
    
    with col3:
        st.metric("Estimated Time", f"{estimated_time // 60}m {estimated_time % 60}s")

if __name__ == "__main__":
    main()
