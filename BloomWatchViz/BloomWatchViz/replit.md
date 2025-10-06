# BloomSphere - NASA Plant Bloom Monitoring System

## Overview

BloomSphere is an interactive web application that leverages NASA satellite imagery to detect, monitor, and predict plant blooming events globally. The system processes multispectral satellite data from MODIS, Landsat, and VIIRS sensors to calculate vegetation indices (NDVI, EVI, SAVI) and provide insights into bloom timing, intensity, and ecological impacts.

The application serves researchers, agricultural professionals, ecologists, and policymakers by providing tools for vegetation monitoring, bloom prediction, species identification, and ecological impact assessment. It supports both global-scale analysis and local region deep-dives with interactive visualizations and data export capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit-based multi-page web application
- **Rationale**: Streamlit provides rapid development of data-centric applications with minimal frontend code, ideal for scientific visualization tools
- **Structure**: Main app.py entry point with modular pages in `/pages` directory following Streamlit's page routing convention
- **Visualization**: Plotly for interactive charts, Folium for mapping, native Streamlit widgets for controls
- **Pros**: Fast iteration, Python-native, excellent for data science workflows
- **Cons**: Limited customization compared to full-stack frameworks, server-side rendering only

**Page Organization**:
- Global mapping with interactive controls
- Data upload and processing for satellite imagery
- Analytics dashboard for trend analysis
- NASA data discovery and download
- Bloom prediction using ML models
- Species identification from spectral signatures
- Comparative temporal/spatial analysis
- Ecological impact assessment

### Backend Architecture

**Processing Pipeline**:
- **Satellite Image Processing** (`utils/satellite_processor.py`): Handles raster data reading, band extraction, and preprocessing for MODIS, Landsat, and VIIRS imagery
- **Vegetation Index Calculation** (`utils/vegetation_indices.py`): Computes NDVI, EVI, SAVI, and other indices from multispectral data
- **Bloom Detection**: Time-series analysis of vegetation indices to identify bloom events
- **ML Predictions** (`utils/bloom_predictor.py`): Uses historical patterns and trend detection for forecasting future bloom events

**Design Pattern**: Service-oriented architecture with utility modules
- **Rationale**: Separates concerns (data processing, calculation, prediction) for maintainability and testability
- **Alternative Considered**: Microservices architecture - rejected due to application scale and deployment complexity

### Data Storage

**Database**: PostgreSQL with PostGIS for spatial data
- **Rationale**: Robust relational database with excellent geospatial support via PostGIS extension
- **Schema**: Stores processed bloom detection results, historical patterns, and location metadata
- **Connection Management**: `database/db_manager.py` handles connections using psycopg2 with context managers
- **Pros**: Mature ecosystem, powerful spatial queries, good performance
- **Cons**: Requires database server management

**Data Files**:
- Sample location data stored as JSON (`data/sample_locations.json`)
- Uploaded satellite imagery processed in-memory or temporary storage
- Export functionality supports CSV, JSON, GeoTIFF, Excel formats

**Alternative Considered**: NoSQL document store - rejected because geospatial query capabilities and structured vegetation index data favor relational model

### External Dependencies

**NASA Earthdata Integration** (`utils/nasa_earthdata.py`):
- **Purpose**: Access satellite imagery from NASA's data repositories
- **APIs**: CMR (Common Metadata Repository), MODIS data via LAADS DAAC, Landsat via USGS, VIIRS data
- **Authentication**: Uses NASA Earthdata credentials (username/password)
- **Collections Supported**:
  - MODIS Terra/Aqua NDVI products (MOD13A2, MYD13A2)
  - Landsat 8/9 Level 2 surface reflectance
  - VIIRS vegetation indices (VNP13A1)

**Satellite Data Providers**:
- **MODIS**: 16-day and 8-day vegetation products at 250m-1km resolution
- **Landsat**: 30m resolution multispectral imagery, extensive archive back to 1970s
- **VIIRS**: Daily vegetation indices at 500m resolution

**Geospatial Libraries**:
- **Rasterio**: Reading and processing geospatial raster data (GeoTIFF, HDF, NetCDF)
- **Folium**: Interactive web mapping with Leaflet.js backend
- **Plotly**: Interactive scientific visualizations and charts

**Scientific Computing Stack**:
- **NumPy**: Array operations for satellite imagery processing
- **Pandas**: Tabular data handling for time-series analysis
- **SciPy**: Signal processing for bloom cycle detection
- **Scikit-learn**: Machine learning models for bloom prediction (LinearRegression, StandardScaler)

**Authentication & Configuration**:
- Environment variables for sensitive credentials (DATABASE_URL, NASA_EARTHDATA_USERNAME, NASA_EARTHDATA_PASSWORD)
- No explicit authentication layer for end users currently implemented

**Key Integration Points**:
1. NASA Earthdata search and download pipeline
2. Database persistence for processed results
3. Real-time satellite image processing and vegetation index calculation
4. Export manager for multi-format data delivery

**Deployment Considerations**:
- Application expects PostgreSQL database (may need to be added if not present)
- NASA Earthdata credentials required for satellite data access
- Large satellite imagery files require adequate storage and memory
- Processing satellite imagery is CPU-intensive (consider optimization or background processing)