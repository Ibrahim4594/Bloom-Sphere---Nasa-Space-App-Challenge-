"""
Database manager for BloomSphere application
Handles connections and operations for PostgreSQL database
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

class DatabaseManager:
    """
    Manages database connections and operations for bloom detection data
    """
    
    def __init__(self):
        """Initialize database connection"""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(self.database_url)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()
    
    def initialize_schema(self):
        """Initialize database schema from SQL file"""
        self.connect()
        
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        try:
            self.cursor.execute(schema_sql)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to initialize schema: {str(e)}")
    
    # Satellite Imagery Operations
    
    def insert_satellite_imagery(self, metadata: Dict) -> int:
        """
        Insert satellite imagery metadata
        
        Args:
            metadata: Dictionary containing imagery metadata
            
        Returns:
            int: ID of inserted record
        """
        self.connect()
        
        query = """
        INSERT INTO satellite_imagery (
            filename, satellite_type, acquisition_date, width, height, bands,
            crs, bounds_north, bounds_south, bounds_east, bounds_west,
            resolution_x, resolution_y, cloud_cover, file_size_mb, processing_status
        ) VALUES (
            %(filename)s, %(satellite_type)s, %(acquisition_date)s, %(width)s, 
            %(height)s, %(bands)s, %(crs)s, %(bounds_north)s, %(bounds_south)s,
            %(bounds_east)s, %(bounds_west)s, %(resolution_x)s, %(resolution_y)s,
            %(cloud_cover)s, %(file_size_mb)s, %(processing_status)s
        ) RETURNING id
        """
        
        self.cursor.execute(query, metadata)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    def update_imagery_status(self, imagery_id: int, status: str):
        """Update processing status of imagery"""
        self.connect()
        
        query = """
        UPDATE satellite_imagery 
        SET processing_status = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """
        
        self.cursor.execute(query, (status, imagery_id))
        self.conn.commit()
    
    def get_satellite_imagery(self, imagery_id: int) -> Optional[Dict]:
        """Retrieve satellite imagery metadata by ID"""
        self.connect()
        
        query = "SELECT * FROM satellite_imagery WHERE id = %s"
        self.cursor.execute(query, (imagery_id,))
        
        return self.cursor.fetchone()
    
    def list_satellite_imagery(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List satellite imagery records"""
        self.connect()
        
        query = """
        SELECT * FROM satellite_imagery 
        ORDER BY upload_date DESC 
        LIMIT %s OFFSET %s
        """
        
        self.cursor.execute(query, (limit, offset))
        return self.cursor.fetchall()
    
    # Vegetation Index Operations
    
    def insert_vegetation_index(self, imagery_id: int, index_type: str, stats: Dict) -> int:
        """
        Insert vegetation index calculation results
        
        Args:
            imagery_id: ID of related satellite imagery
            index_type: Type of vegetation index (NDVI, EVI, etc.)
            stats: Dictionary containing statistical values
            
        Returns:
            int: ID of inserted record
        """
        self.connect()
        
        query = """
        INSERT INTO vegetation_indices (
            imagery_id, index_type, mean_value, median_value, min_value, max_value,
            std_dev, percentile_25, percentile_75, percentile_90, valid_pixels, total_pixels
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            imagery_id, index_type,
            stats.get('mean'), stats.get('median'), stats.get('min'), stats.get('max'),
            stats.get('std'), stats.get('percentile_25'), stats.get('percentile_75'),
            stats.get('percentile_90'), stats.get('valid_pixels'), stats.get('total_pixels')
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    def get_vegetation_indices(self, imagery_id: int) -> List[Dict]:
        """Get all vegetation indices for an imagery"""
        self.connect()
        
        query = "SELECT * FROM vegetation_indices WHERE imagery_id = %s"
        self.cursor.execute(query, (imagery_id,))
        
        return self.cursor.fetchall()
    
    # Bloom Detection Operations
    
    def insert_bloom_detection(self, imagery_id: int, detection_params: Dict, 
                               detection_stats: Dict) -> int:
        """
        Insert bloom detection results
        
        Args:
            imagery_id: ID of related satellite imagery
            detection_params: Parameters used for detection
            detection_stats: Resulting statistics
            
        Returns:
            int: ID of inserted record
        """
        self.connect()
        
        query = """
        INSERT INTO bloom_detections (
            imagery_id, bloom_threshold, detection_method, total_bloom_pixels,
            bloom_coverage_percent, bloom_clusters, mean_bloom_intensity,
            max_bloom_intensity, bloom_area_km2, confidence_score
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            imagery_id,
            detection_params.get('threshold'),
            detection_params.get('method'),
            detection_stats.get('bloom_pixels'),
            detection_stats.get('bloom_coverage_percent'),
            detection_stats.get('total_bloom_clusters'),
            detection_stats.get('mean_bloom_intensity'),
            detection_stats.get('max_bloom_intensity'),
            detection_stats.get('bloom_area_km2'),
            detection_params.get('confidence_score', 0.85)
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    def insert_bloom_locations(self, detection_id: int, locations: List[Dict]):
        """
        Insert multiple bloom location points
        
        Args:
            detection_id: ID of bloom detection
            locations: List of location dictionaries
        """
        self.connect()
        
        query = """
        INSERT INTO bloom_locations (
            detection_id, latitude, longitude, intensity, confidence,
            land_cover, bloom_stage, ecosystem_type
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        values_list = [
            (
                detection_id,
                loc.get('lat'),
                loc.get('lon'),
                loc.get('intensity'),
                loc.get('confidence', 0.8),
                loc.get('land_cover'),
                loc.get('bloom_stage'),
                loc.get('ecosystem_type')
            )
            for loc in locations
        ]
        
        self.cursor.executemany(query, values_list)
        self.conn.commit()
    
    def get_bloom_locations(self, detection_id: int) -> List[Dict]:
        """Get bloom locations for a detection"""
        self.connect()
        
        query = "SELECT * FROM bloom_locations WHERE detection_id = %s"
        self.cursor.execute(query, (detection_id,))
        
        return self.cursor.fetchall()
    
    def get_bloom_locations_by_region(self, north: float, south: float, 
                                     east: float, west: float,
                                     start_date: Optional[date] = None,
                                     end_date: Optional[date] = None) -> List[Dict]:
        """
        Get bloom locations within geographic bounds and optional date range
        
        Args:
            north, south, east, west: Geographic boundaries
            start_date, end_date: Optional date range filter
            
        Returns:
            List of bloom location records
        """
        self.connect()
        
        query = """
        SELECT bl.*, bd.detection_date, bd.bloom_threshold
        FROM bloom_locations bl
        JOIN bloom_detections bd ON bl.detection_id = bd.id
        WHERE bl.latitude BETWEEN %s AND %s
        AND bl.longitude BETWEEN %s AND %s
        """
        
        params = [south, north, west, east]
        
        if start_date and end_date:
            query += " AND bd.detection_date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        
        query += " ORDER BY bd.detection_date DESC"
        
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    # Temporal Analysis Operations
    
    def insert_temporal_analysis(self, analysis_data: Dict) -> int:
        """Insert temporal bloom analysis results"""
        self.connect()
        
        query = """
        INSERT INTO temporal_analysis (
            location_name, latitude, longitude, start_date, end_date,
            peak_bloom_date, peak_bloom_coverage, average_intensity,
            bloom_duration_days, trend_direction, trend_value
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            analysis_data.get('location_name'),
            analysis_data.get('latitude'),
            analysis_data.get('longitude'),
            analysis_data.get('start_date'),
            analysis_data.get('end_date'),
            analysis_data.get('peak_bloom_date'),
            analysis_data.get('peak_bloom_coverage'),
            analysis_data.get('average_intensity'),
            analysis_data.get('bloom_duration_days'),
            analysis_data.get('trend_direction'),
            analysis_data.get('trend_value')
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    # Species Prediction Operations
    
    def insert_species_prediction(self, bloom_location_id: int, prediction: Dict) -> int:
        """Insert species prediction for a bloom location"""
        self.connect()
        
        query = """
        INSERT INTO species_predictions (
            bloom_location_id, species_name, confidence, spectral_signature,
            vegetation_type, additional_characteristics
        ) VALUES (
            %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            bloom_location_id,
            prediction.get('species_name'),
            prediction.get('confidence'),
            Json(prediction.get('spectral_signature', {})),
            prediction.get('vegetation_type'),
            Json(prediction.get('additional_characteristics', {}))
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    # Ecological Impact Operations
    
    def insert_ecological_impact(self, detection_id: int, impact: Dict) -> int:
        """Insert ecological impact annotation"""
        self.connect()
        
        query = """
        INSERT INTO ecological_impacts (
            detection_id, impact_type, impact_level, description,
            quantitative_value, unit, forecast_period
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            detection_id,
            impact.get('impact_type'),
            impact.get('impact_level'),
            impact.get('description'),
            impact.get('quantitative_value'),
            impact.get('unit'),
            impact.get('forecast_period')
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    def get_ecological_impacts(self, detection_id: int) -> List[Dict]:
        """Get ecological impacts for a detection"""
        self.connect()
        
        query = "SELECT * FROM ecological_impacts WHERE detection_id = %s"
        self.cursor.execute(query, (detection_id,))
        
        return self.cursor.fetchall()
    
    # Export History Operations
    
    def insert_export_record(self, export_data: Dict) -> int:
        """Record an export operation"""
        self.connect()
        
        query = """
        INSERT INTO export_history (
            export_type, filename, imagery_ids, detection_ids,
            date_range_start, date_range_end, region, file_size_mb, exported_by
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
        """
        
        values = (
            export_data.get('export_type'),
            export_data.get('filename'),
            export_data.get('imagery_ids', []),
            export_data.get('detection_ids', []),
            export_data.get('date_range_start'),
            export_data.get('date_range_end'),
            export_data.get('region'),
            export_data.get('file_size_mb'),
            export_data.get('exported_by', 'user')
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        
        return self.cursor.fetchone()['id']
    
    # Summary and Analytics Queries
    
    def get_bloom_summary(self, limit: int = 100) -> pd.DataFrame:
        """Get bloom summary view as DataFrame"""
        self.connect()
        
        query = "SELECT * FROM bloom_summary ORDER BY acquisition_date DESC LIMIT %s"
        self.cursor.execute(query, (limit,))
        
        results = self.cursor.fetchall()
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def get_temporal_trends(self, location_name: str = None, 
                           start_date: date = None, 
                           end_date: date = None) -> pd.DataFrame:
        """Get temporal trend data"""
        self.connect()
        
        query = "SELECT * FROM temporal_analysis WHERE 1=1"
        params = []
        
        if location_name:
            query += " AND location_name = %s"
            params.append(location_name)
        
        if start_date:
            query += " AND start_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND end_date <= %s"
            params.append(end_date)
        
        query += " ORDER BY start_date"
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Get overall database statistics"""
        self.connect()
        
        stats = {}
        
        # Total imagery count
        self.cursor.execute("SELECT COUNT(*) as count FROM satellite_imagery")
        stats['total_imagery'] = self.cursor.fetchone()['count']
        
        # Total bloom detections
        self.cursor.execute("SELECT COUNT(*) as count FROM bloom_detections")
        stats['total_detections'] = self.cursor.fetchone()['count']
        
        # Total bloom locations
        self.cursor.execute("SELECT COUNT(*) as count FROM bloom_locations")
        stats['total_bloom_locations'] = self.cursor.fetchone()['count']
        
        # Average bloom coverage
        self.cursor.execute("""
            SELECT AVG(bloom_coverage_percent) as avg_coverage 
            FROM bloom_detections
        """)
        result = self.cursor.fetchone()
        stats['avg_bloom_coverage'] = result['avg_coverage'] if result['avg_coverage'] else 0
        
        # Date range
        self.cursor.execute("""
            SELECT MIN(acquisition_date) as min_date, MAX(acquisition_date) as max_date
            FROM satellite_imagery
        """)
        dates = self.cursor.fetchone()
        stats['date_range'] = {
            'start': dates['min_date'],
            'end': dates['max_date']
        }
        
        return stats
