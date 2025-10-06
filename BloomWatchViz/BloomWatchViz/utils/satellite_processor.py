import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

class SatelliteProcessor:
    """
    A class for processing satellite imagery for bloom detection.
    Handles reading, preprocessing, and analyzing satellite data from various sources.
    """
    
    def __init__(self):
        self.supported_formats = ['.tif', '.tiff', '.hdf', '.nc']
        self.band_configurations = {
            'MODIS': {
                'red': 0,      # Band 1 (620-670 nm)
                'nir': 1,      # Band 2 (841-876 nm)
                'blue': 2,     # Band 3 (459-479 nm)
                'green': 3     # Band 4 (545-565 nm)
            },
            'LANDSAT': {
                'blue': 0,     # Band 2 (450-515 nm)
                'green': 1,    # Band 3 (525-600 nm)
                'red': 2,      # Band 4 (630-680 nm)
                'nir': 3,      # Band 5 (845-885 nm)
                'swir1': 4,    # Band 6 (1560-1660 nm)
                'swir2': 5     # Band 7 (2100-2300 nm)
            },
            'VIIRS': {
                'red': 0,      # I1 (600-680 nm)
                'nir': 1,      # I2 (845-885 nm)
                'blue': 2,     # M3 (478-498 nm)
                'green': 3     # M4 (545-565 nm)
            }
        }
    
    def read_imagery(self, file_path: str, satellite_type: str = 'MODIS') -> np.ndarray:
        """
        Read satellite imagery from file.
        
        Args:
            file_path (str): Path to the satellite imagery file
            satellite_type (str): Type of satellite data ('MODIS', 'LANDSAT', 'VIIRS')
            
        Returns:
            np.ndarray: Multi-band imagery data with shape (bands, height, width)
        """
        
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                imagery = src.read()
                
                # Handle nodata values
                if src.nodata is not None:
                    imagery = np.where(imagery == src.nodata, np.nan, imagery)
                
                # Store metadata for later use
                self.metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'height': src.height,
                    'width': src.width,
                    'count': src.count,
                    'dtype': src.dtype,
                    'bounds': src.bounds
                }
                
                return self._preprocess_imagery(imagery, satellite_type)
                
        except Exception as e:
            raise Exception(f"Error reading imagery from {file_path}: {str(e)}")
    
    def _preprocess_imagery(self, imagery: np.ndarray, satellite_type: str) -> np.ndarray:
        """
        Preprocess satellite imagery data.
        
        Args:
            imagery (np.ndarray): Raw imagery data
            satellite_type (str): Type of satellite data
            
        Returns:
            np.ndarray: Preprocessed imagery data
        """
        
        # Scale values to reflectance (0-1) if needed
        if imagery.max() > 1.0:
            # Assume values are in the range 0-10000 (common for satellite data)
            imagery = imagery / 10000.0
        
        # Clip to valid reflectance range
        imagery = np.clip(imagery, 0, 1)
        
        # Apply basic noise reduction (median filter on each band)
        from scipy import ndimage
        
        processed_imagery = np.zeros_like(imagery)
        for band_idx in range(imagery.shape[0]):
            processed_imagery[band_idx] = ndimage.median_filter(
                imagery[band_idx], size=3
            )
        
        return processed_imagery
    
    def detect_blooms(self, vegetation_index: np.ndarray, 
                     threshold: float = 0.4,
                     min_area_pixels: int = 10) -> np.ndarray:
        """
        Detect bloom events from vegetation index data.
        
        Args:
            vegetation_index (np.ndarray): Vegetation index values (e.g., NDVI)
            threshold (float): Minimum vegetation index value to consider as bloom
            min_area_pixels (int): Minimum area in pixels for a valid bloom detection
            
        Returns:
            np.ndarray: Binary mask indicating bloom locations
        """
        
        # Basic threshold-based bloom detection
        bloom_mask = vegetation_index > threshold
        
        # Remove small isolated pixels using morphological operations
        from scipy import ndimage
        
        # Close small gaps
        bloom_mask = ndimage.binary_closing(bloom_mask, structure=np.ones((3, 3)))
        
        # Remove small objects
        bloom_mask = self._remove_small_objects(bloom_mask, min_area_pixels)
        
        return bloom_mask.astype(np.uint8)
    
    def _remove_small_objects(self, binary_mask: np.ndarray, 
                            min_size: int) -> np.ndarray:
        """
        Remove small connected components from binary mask.
        
        Args:
            binary_mask (np.ndarray): Binary mask
            min_size (int): Minimum size of objects to keep
            
        Returns:
            np.ndarray: Filtered binary mask
        """
        
        from scipy import ndimage
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        # Count pixels in each component
        component_sizes = ndimage.sum(binary_mask, labeled_array, 
                                    range(num_features + 1))
        
        # Create mask for components larger than minimum size
        size_mask = component_sizes >= min_size
        size_mask[0] = 0  # Background should remain 0
        
        # Apply size filter
        filtered_mask = size_mask[labeled_array]
        
        return filtered_mask
    
    def calculate_bloom_statistics(self, bloom_mask: np.ndarray, 
                                 vegetation_index: np.ndarray) -> dict:
        """
        Calculate statistics for detected bloom areas.
        
        Args:
            bloom_mask (np.ndarray): Binary mask of bloom locations
            vegetation_index (np.ndarray): Vegetation index values
            
        Returns:
            dict: Dictionary containing bloom statistics
        """
        
        bloom_pixels = np.sum(bloom_mask)
        total_valid_pixels = np.sum(~np.isnan(vegetation_index))
        
        if bloom_pixels == 0:
            return {
                'bloom_coverage_percent': 0.0,
                'bloom_pixels': 0,
                'mean_bloom_intensity': 0.0,
                'max_bloom_intensity': 0.0,
                'total_bloom_clusters': 0
            }
        
        # Calculate coverage percentage
        coverage_percent = (bloom_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
        
        # Calculate intensity statistics for bloom areas
        bloom_values = vegetation_index[bloom_mask.astype(bool)]
        mean_intensity = np.nanmean(bloom_values) if len(bloom_values) > 0 else 0
        max_intensity = np.nanmax(bloom_values) if len(bloom_values) > 0 else 0
        
        # Count number of bloom clusters
        from scipy import ndimage
        labeled_blooms, num_clusters = ndimage.label(bloom_mask)
        
        statistics = {
            'bloom_coverage_percent': coverage_percent,
            'bloom_pixels': int(bloom_pixels),
            'mean_bloom_intensity': float(mean_intensity),
            'max_bloom_intensity': float(max_intensity),
            'total_bloom_clusters': int(num_clusters),
            'bloom_area_km2': self._pixels_to_km2(bloom_pixels) if hasattr(self, 'metadata') else None
        }
        
        return statistics
    
    def _pixels_to_km2(self, num_pixels: int) -> float:
        """
        Convert pixel count to area in square kilometers.
        
        Args:
            num_pixels (int): Number of pixels
            
        Returns:
            float: Area in square kilometers
        """
        
        if not hasattr(self, 'metadata'):
            return None
        
        # Get pixel size from transform (assuming square pixels)
        pixel_width = abs(self.metadata['transform'][0])  # degrees
        pixel_height = abs(self.metadata['transform'][4])  # degrees
        
        # Convert to km² (rough approximation)
        # 1 degree ≈ 111.32 km at equator
        pixel_area_km2 = (pixel_width * 111.32) * (pixel_height * 111.32)
        
        return num_pixels * pixel_area_km2
    
    def temporal_bloom_analysis(self, time_series_data: list, 
                              dates: list) -> dict:
        """
        Analyze bloom patterns across multiple time points.
        
        Args:
            time_series_data (list): List of vegetation index arrays for different dates
            dates (list): List of corresponding dates
            
        Returns:
            dict: Temporal bloom analysis results
        """
        
        if len(time_series_data) != len(dates):
            raise ValueError("Number of data arrays must match number of dates")
        
        bloom_timeline = []
        intensity_timeline = []
        
        for i, (data, date) in enumerate(zip(time_series_data, dates)):
            # Detect blooms for this time point
            bloom_mask = self.detect_blooms(data)
            stats = self.calculate_bloom_statistics(bloom_mask, data)
            
            bloom_timeline.append({
                'date': date,
                'bloom_coverage': stats['bloom_coverage_percent'],
                'mean_intensity': stats['mean_bloom_intensity'],
                'bloom_clusters': stats['total_bloom_clusters']
            })
            
            intensity_timeline.append(np.nanmean(data))
        
        # Calculate temporal trends
        coverage_values = [point['bloom_coverage'] for point in bloom_timeline]
        intensity_values = [point['mean_intensity'] for point in bloom_timeline]
        
        # Find peak bloom period
        peak_index = np.argmax(coverage_values) if coverage_values else 0
        peak_date = dates[peak_index] if dates else None
        
        # Calculate trend (simple linear regression)
        if len(coverage_values) > 1:
            x = np.arange(len(coverage_values))
            coverage_trend = np.polyfit(x, coverage_values, 1)[0]  # Slope
            intensity_trend = np.polyfit(x, intensity_values, 1)[0]  # Slope
        else:
            coverage_trend = 0
            intensity_trend = 0
        
        return {
            'bloom_timeline': bloom_timeline,
            'peak_bloom_date': peak_date,
            'peak_bloom_coverage': max(coverage_values) if coverage_values else 0,
            'coverage_trend': coverage_trend,
            'intensity_trend': intensity_trend,
            'bloom_duration_estimate': self._estimate_bloom_duration(coverage_values)
        }
    
    def _estimate_bloom_duration(self, coverage_timeline: list) -> int:
        """
        Estimate bloom duration based on coverage timeline.
        
        Args:
            coverage_timeline (list): Timeline of bloom coverage percentages
            
        Returns:
            int: Estimated bloom duration in time periods
        """
        
        if not coverage_timeline:
            return 0
        
        # Find start and end of significant bloom activity
        threshold = max(coverage_timeline) * 0.2 if max(coverage_timeline) > 0 else 0
        
        significant_indices = [i for i, val in enumerate(coverage_timeline) 
                             if val > threshold]
        
        if not significant_indices:
            return 0
        
        duration = significant_indices[-1] - significant_indices[0] + 1
        return duration
    
    def export_results(self, bloom_mask: np.ndarray, 
                      output_path: str,
                      format_type: str = 'geotiff') -> None:
        """
        Export bloom detection results to file.
        
        Args:
            bloom_mask (np.ndarray): Binary bloom mask
            output_path (str): Output file path
            format_type (str): Output format ('geotiff', 'numpy', 'csv')
        """
        
        try:
            if format_type.lower() == 'geotiff':
                self._export_geotiff(bloom_mask, output_path)
            elif format_type.lower() == 'numpy':
                np.save(output_path, bloom_mask)
            elif format_type.lower() == 'csv':
                # Export as coordinates of bloom pixels
                bloom_coords = np.where(bloom_mask)
                coords_df = pd.DataFrame({
                    'row': bloom_coords[0],
                    'col': bloom_coords[1],
                    'bloom_detected': 1
                })
                coords_df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Error exporting results: {str(e)}")
    
    def _export_geotiff(self, data: np.ndarray, output_path: str) -> None:
        """
        Export data as GeoTIFF file.
        
        Args:
            data (np.ndarray): Data to export
            output_path (str): Output file path
        """
        
        if not hasattr(self, 'metadata'):
            raise ValueError("No metadata available. Read imagery first.")
        
        # Ensure data is 2D
        if data.ndim == 3:
            data = data[0]  # Take first band if 3D
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=self.metadata['crs'],
            transform=self.metadata['transform'],
            compress='lzw'
        ) as dst:
            dst.write(data, 1)
