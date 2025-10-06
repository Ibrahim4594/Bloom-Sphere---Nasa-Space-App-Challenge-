"""
Export Manager for BloomSphere application
Handles export of bloom detection data to various formats
"""

import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional
import io
import rasterio
from rasterio.transform import from_bounds

class ExportManager:
    """
    Manages export operations for bloom detection data
    """
    
    def __init__(self):
        """Initialize export manager"""
        self.supported_formats = ['csv', 'json', 'geotiff', 'excel']
    
    def export_to_csv(self, data: pd.DataFrame, filename: str = None) -> tuple:
        """
        Export data to CSV format
        
        Args:
            data: DataFrame to export
            filename: Optional filename
            
        Returns:
            tuple: (csv_string, filename)
        """
        if filename is None:
            filename = f"bloom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        csv_string = data.to_csv(index=False)
        
        return csv_string, filename
    
    def export_to_json(self, data: pd.DataFrame, filename: str = None) -> tuple:
        """
        Export data to JSON format
        
        Args:
            data: DataFrame to export
            filename: Optional filename
            
        Returns:
            tuple: (json_string, filename)
        """
        if filename is None:
            filename = f"bloom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert DataFrame to records format
        records = data.to_dict(orient='records')
        
        # Create structured JSON
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_records': len(records),
                'data_source': 'BloomSphere - NASA Satellite Data Analysis'
            },
            'data': records
        }
        
        json_string = json.dumps(export_data, indent=2, default=str)
        
        return json_string, filename
    
    def export_to_excel(self, data: pd.DataFrame, filename: str = None) -> tuple:
        """
        Export data to Excel format
        
        Args:
            data: DataFrame to export
            filename: Optional filename
            
        Returns:
            tuple: (excel_bytes, filename)
        """
        if filename is None:
            filename = f"bloom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Bloom Data')
        
        excel_bytes = output.getvalue()
        
        return excel_bytes, filename
    
    def export_bloom_locations_to_geojson(self, locations: List[Dict], 
                                         filename: str = None) -> tuple:
        """
        Export bloom locations to GeoJSON format
        
        Args:
            locations: List of bloom location dictionaries
            filename: Optional filename
            
        Returns:
            tuple: (geojson_string, filename)
        """
        if filename is None:
            filename = f"bloom_locations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        
        # Create GeoJSON structure
        features = []
        
        for loc in locations:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [loc.get('lon', loc.get('longitude', 0)), 
                                  loc.get('lat', loc.get('latitude', 0))]
                },
                'properties': {
                    k: v for k, v in loc.items() 
                    if k not in ['lat', 'latitude', 'lon', 'longitude']
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_features': len(features),
                'source': 'BloomSphere'
            }
        }
        
        geojson_string = json.dumps(geojson, indent=2, default=str)
        
        return geojson_string, filename
    
    def export_raster_to_geotiff(self, raster_data: np.ndarray, 
                                 bounds: tuple, crs: str = 'EPSG:4326',
                                 filename: str = None) -> tuple:
        """
        Export raster data (vegetation index, bloom mask) to GeoTIFF
        
        Args:
            raster_data: 2D numpy array of raster data
            bounds: Tuple of (west, south, east, north) coordinates
            crs: Coordinate reference system
            filename: Optional filename
            
        Returns:
            tuple: (bytes, filename)
        """
        if filename is None:
            filename = f"bloom_raster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        
        # Ensure data is 2D
        if raster_data.ndim > 2:
            raster_data = raster_data[0]
        
        height, width = raster_data.shape
        
        # Create transform
        transform = from_bounds(*bounds, width, height)
        
        # Create in-memory file
        memfile = io.BytesIO()
        
        with rasterio.MemoryFile(memfile) as memf:
            with memf.open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=raster_data.dtype,
                crs=crs,
                transform=transform,
                compress='lzw'
            ) as dataset:
                dataset.write(raster_data, 1)
        
        return memfile.getvalue(), filename
    
    def create_bloom_report(self, summary_stats: Dict, 
                           bloom_data: pd.DataFrame = None,
                           region: str = None) -> tuple:
        """
        Create a comprehensive bloom report in text format
        
        Args:
            summary_stats: Dictionary of summary statistics
            bloom_data: Optional DataFrame with detailed bloom data
            region: Optional region name
            
        Returns:
            tuple: (report_text, filename)
        """
        filename = f"bloom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BLOOMWATCH - PLANT BLOOM MONITORING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if region:
            report_lines.append(f"Region: {region}")
        
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        
        for key, value in summary_stats.items():
            formatted_key = key.replace('_', ' ').title()
            report_lines.append(f"{formatted_key:.<40} {value}")
        
        if bloom_data is not None and not bloom_data.empty:
            report_lines.append("")
            report_lines.append("-" * 80)
            report_lines.append("DETAILED BLOOM DATA")
            report_lines.append("-" * 80)
            report_lines.append("")
            report_lines.append(bloom_data.to_string())
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        return report_text, filename
    
    def export_bloom_summary(self, imagery_list: List[Dict],
                            detection_list: List[Dict],
                            format: str = 'csv') -> tuple:
        """
        Export summary of bloom detections across multiple imagery
        
        Args:
            imagery_list: List of imagery metadata dictionaries
            detection_list: List of bloom detection dictionaries
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            tuple: (export_data, filename)
        """
        # Create summary DataFrame
        summary_data = []
        
        for imagery, detection in zip(imagery_list, detection_list):
            summary_data.append({
                'Date': imagery.get('acquisition_date', 'N/A'),
                'Filename': imagery.get('filename', 'N/A'),
                'Satellite': imagery.get('satellite_type', 'N/A'),
                'Coverage Area (km²)': detection.get('bloom_area_km2', 0),
                'Bloom Coverage (%)': detection.get('bloom_coverage_percent', 0),
                'Mean Intensity': detection.get('mean_bloom_intensity', 0),
                'Bloom Clusters': detection.get('bloom_clusters', 0),
                'Confidence': detection.get('confidence_score', 0)
            })
        
        df = pd.DataFrame(summary_data)
        
        if format == 'csv':
            return self.export_to_csv(df)
        elif format == 'json':
            return self.export_to_json(df)
        elif format == 'excel':
            return self.export_to_excel(df)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_file_size_mb(self, data: bytes) -> float:
        """Calculate file size in MB"""
        return len(data) / (1024 * 1024)
