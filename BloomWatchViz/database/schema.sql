-- BloomSphere Database Schema
-- Schema for storing satellite imagery metadata, vegetation indices, and bloom detections

-- Table for storing uploaded satellite imagery metadata
CREATE TABLE IF NOT EXISTS satellite_imagery (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    satellite_type VARCHAR(50),  -- MODIS, Landsat, VIIRS
    acquisition_date DATE,
    width INTEGER,
    height INTEGER,
    bands INTEGER,
    crs VARCHAR(100),
    bounds_north FLOAT,
    bounds_south FLOAT,
    bounds_east FLOAT,
    bounds_west FLOAT,
    resolution_x FLOAT,
    resolution_y FLOAT,
    cloud_cover FLOAT,
    file_size_mb FLOAT,
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing vegetation index results
CREATE TABLE IF NOT EXISTS vegetation_indices (
    id SERIAL PRIMARY KEY,
    imagery_id INTEGER REFERENCES satellite_imagery(id) ON DELETE CASCADE,
    index_type VARCHAR(20) NOT NULL,  -- NDVI, EVI, SAVI, ARVI
    mean_value FLOAT,
    median_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    std_dev FLOAT,
    percentile_25 FLOAT,
    percentile_75 FLOAT,
    percentile_90 FLOAT,
    valid_pixels INTEGER,
    total_pixels INTEGER,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for bloom detections
CREATE TABLE IF NOT EXISTS bloom_detections (
    id SERIAL PRIMARY KEY,
    imagery_id INTEGER REFERENCES satellite_imagery(id) ON DELETE CASCADE,
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bloom_threshold FLOAT,
    detection_method VARCHAR(50),  -- threshold, percentile, adaptive
    total_bloom_pixels INTEGER,
    bloom_coverage_percent FLOAT,
    bloom_clusters INTEGER,
    mean_bloom_intensity FLOAT,
    max_bloom_intensity FLOAT,
    bloom_area_km2 FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for individual bloom locations/points
CREATE TABLE IF NOT EXISTS bloom_locations (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES bloom_detections(id) ON DELETE CASCADE,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    intensity FLOAT,
    confidence FLOAT,
    land_cover VARCHAR(100),
    bloom_stage VARCHAR(50),  -- pre-bloom, early bloom, peak bloom, late bloom, post-bloom
    ecosystem_type VARCHAR(100),
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_latitude CHECK (latitude >= -90 AND latitude <= 90),
    CONSTRAINT valid_longitude CHECK (longitude >= -180 AND longitude <= 180)
);

-- Table for temporal bloom analysis
CREATE TABLE IF NOT EXISTS temporal_analysis (
    id SERIAL PRIMARY KEY,
    location_name VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    start_date DATE,
    end_date DATE,
    peak_bloom_date DATE,
    peak_bloom_coverage FLOAT,
    average_intensity FLOAT,
    bloom_duration_days INTEGER,
    trend_direction VARCHAR(20),  -- increasing, decreasing, stable
    trend_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for species predictions
CREATE TABLE IF NOT EXISTS species_predictions (
    id SERIAL PRIMARY KEY,
    bloom_location_id INTEGER REFERENCES bloom_locations(id) ON DELETE CASCADE,
    species_name VARCHAR(255),
    confidence FLOAT,
    spectral_signature JSONB,  -- Store spectral characteristics
    vegetation_type VARCHAR(100),
    additional_characteristics JSONB,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for ecological impact annotations
CREATE TABLE IF NOT EXISTS ecological_impacts (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES bloom_detections(id) ON DELETE CASCADE,
    impact_type VARCHAR(100),  -- pollinator_activity, crop_yield, pollen_forecast, etc.
    impact_level VARCHAR(50),  -- low, moderate, high, critical
    description TEXT,
    quantitative_value FLOAT,
    unit VARCHAR(50),
    forecast_period VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for exported reports
CREATE TABLE IF NOT EXISTS export_history (
    id SERIAL PRIMARY KEY,
    export_type VARCHAR(50),  -- csv, geotiff, json, report
    filename VARCHAR(255),
    imagery_ids INTEGER[],
    detection_ids INTEGER[],
    date_range_start DATE,
    date_range_end DATE,
    region VARCHAR(100),
    file_size_mb FLOAT,
    exported_by VARCHAR(100),
    export_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_satellite_imagery_date ON satellite_imagery(acquisition_date);
CREATE INDEX IF NOT EXISTS idx_satellite_imagery_type ON satellite_imagery(satellite_type);
CREATE INDEX IF NOT EXISTS idx_bloom_locations_coords ON bloom_locations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_bloom_locations_date ON bloom_locations(detection_date);
CREATE INDEX IF NOT EXISTS idx_temporal_analysis_dates ON temporal_analysis(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_vegetation_indices_imagery ON vegetation_indices(imagery_id);
CREATE INDEX IF NOT EXISTS idx_bloom_detections_imagery ON bloom_detections(imagery_id);

-- Create view for easy bloom summary queries
CREATE OR REPLACE VIEW bloom_summary AS
SELECT 
    si.id as imagery_id,
    si.filename,
    si.satellite_type,
    si.acquisition_date,
    bd.bloom_coverage_percent,
    bd.bloom_clusters,
    bd.mean_bloom_intensity,
    bd.bloom_area_km2,
    COUNT(bl.id) as bloom_points,
    vi_ndvi.mean_value as ndvi_mean,
    vi_evi.mean_value as evi_mean
FROM satellite_imagery si
LEFT JOIN bloom_detections bd ON si.id = bd.imagery_id
LEFT JOIN bloom_locations bl ON bd.id = bl.detection_id
LEFT JOIN vegetation_indices vi_ndvi ON si.id = vi_ndvi.imagery_id AND vi_ndvi.index_type = 'NDVI'
LEFT JOIN vegetation_indices vi_evi ON si.id = vi_evi.imagery_id AND vi_evi.index_type = 'EVI'
GROUP BY si.id, si.filename, si.satellite_type, si.acquisition_date, 
         bd.bloom_coverage_percent, bd.bloom_clusters, bd.mean_bloom_intensity, 
         bd.bloom_area_km2, vi_ndvi.mean_value, vi_evi.mean_value;
