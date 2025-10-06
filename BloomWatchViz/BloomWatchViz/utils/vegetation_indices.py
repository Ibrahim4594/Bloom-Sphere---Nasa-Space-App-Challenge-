import numpy as np
from typing import Union, Optional
import warnings

class VegetationIndices:
    """
    A class for calculating various vegetation indices from satellite imagery.
    Supports NDVI, EVI, SAVI, ARVI and other common vegetation indices used in
    remote sensing applications for bloom detection and vegetation monitoring.
    """
    
    def __init__(self):
        """Initialize the VegetationIndices calculator."""
        self.supported_indices = [
            'NDVI', 'EVI', 'SAVI', 'ARVI', 'GNDVI', 'NDRE', 'CVI', 'MCARI'
        ]
        
        # Default parameters for various indices
        self.default_params = {
            'EVI': {'L': 1.0, 'C1': 6.0, 'C2': 7.5, 'G': 2.5},
            'SAVI': {'L': 0.5},
            'ARVI': {'gamma': 1.0}
        }
    
    def calculate_ndvi(self, imagery: np.ndarray, 
                      red_band: int = 0, nir_band: int = 3) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery with shape (bands, height, width)
            red_band (int): Index of the red band (typically 630-680 nm)
            nir_band (int): Index of the near-infrared band (typically 845-885 nm)
            
        Returns:
            np.ndarray: NDVI values ranging from -1 to 1
        """
        
        try:
            # Extract bands
            red = imagery[red_band].astype(np.float64)
            nir = imagery[nir_band].astype(np.float64)
            
            # Calculate NDVI
            numerator = nir - red
            denominator = nir + red
            
            # Handle division by zero
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ndvi = np.divide(numerator, denominator, 
                               out=np.zeros_like(numerator), 
                               where=(denominator != 0))
            
            # Set invalid values to NaN
            ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
            
            # Clip to valid NDVI range
            ndvi = np.clip(ndvi, -1, 1)
            
            return ndvi
            
        except Exception as e:
            raise Exception(f"Error calculating NDVI: {str(e)}")
    
    def calculate_evi(self, imagery: np.ndarray,
                     red_band: int = 0, nir_band: int = 3, blue_band: int = 2,
                     L: float = None, C1: float = None, C2: float = None, 
                     G: float = None) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        EVI = G * [(NIR - Red) / (NIR + C1*Red - C2*Blue + L)]
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            red_band (int): Index of the red band
            nir_band (int): Index of the near-infrared band
            blue_band (int): Index of the blue band
            L (float): Canopy background adjustment factor (default: 1.0)
            C1 (float): Aerosol resistance coefficient for red band (default: 6.0)
            C2 (float): Aerosol resistance coefficient for blue band (default: 7.5)
            G (float): Gain factor (default: 2.5)
            
        Returns:
            np.ndarray: EVI values typically ranging from -1 to 1
        """
        
        # Use default parameters if not provided
        params = self.default_params['EVI']
        L = L if L is not None else params['L']
        C1 = C1 if C1 is not None else params['C1']
        C2 = C2 if C2 is not None else params['C2']
        G = G if G is not None else params['G']
        
        try:
            # Extract bands
            red = imagery[red_band].astype(np.float64)
            nir = imagery[nir_band].astype(np.float64)
            blue = imagery[blue_band].astype(np.float64)
            
            # Calculate EVI components
            numerator = nir - red
            denominator = nir + C1 * red - C2 * blue + L
            
            # Calculate EVI
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                evi = G * np.divide(numerator, denominator,
                                  out=np.zeros_like(numerator),
                                  where=(denominator != 0))
            
            # Set invalid values to NaN
            evi = np.where(np.isfinite(evi), evi, np.nan)
            
            # Clip to reasonable range
            evi = np.clip(evi, -1, 1)
            
            return evi
            
        except Exception as e:
            raise Exception(f"Error calculating EVI: {str(e)}")
    
    def calculate_savi(self, imagery: np.ndarray,
                      red_band: int = 0, nir_band: int = 3,
                      L: float = None) -> np.ndarray:
        """
        Calculate Soil-Adjusted Vegetation Index (SAVI).
        
        SAVI = [(NIR - Red) / (NIR + Red + L)] * (1 + L)
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            red_band (int): Index of the red band
            nir_band (int): Index of the near-infrared band
            L (float): Soil brightness correction factor (default: 0.5)
                      L=0 for very high vegetation cover
                      L=1 for very low vegetation cover
                      L=0.5 for intermediate vegetation cover
            
        Returns:
            np.ndarray: SAVI values typically ranging from -1 to 1
        """
        
        # Use default parameter if not provided
        L = L if L is not None else self.default_params['SAVI']['L']
        
        try:
            # Extract bands
            red = imagery[red_band].astype(np.float64)
            nir = imagery[nir_band].astype(np.float64)
            
            # Calculate SAVI
            numerator = nir - red
            denominator = nir + red + L
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                savi = (1 + L) * np.divide(numerator, denominator,
                                         out=np.zeros_like(numerator),
                                         where=(denominator != 0))
            
            # Set invalid values to NaN
            savi = np.where(np.isfinite(savi), savi, np.nan)
            
            # Clip to reasonable range
            savi = np.clip(savi, -1, 1)
            
            return savi
            
        except Exception as e:
            raise Exception(f"Error calculating SAVI: {str(e)}")
    
    def calculate_arvi(self, imagery: np.ndarray,
                      red_band: int = 0, nir_band: int = 3, blue_band: int = 2,
                      gamma: float = None) -> np.ndarray:
        """
        Calculate Atmospherically Resistant Vegetation Index (ARVI).
        
        ARVI = (NIR - RB) / (NIR + RB)
        where RB = Red - gamma * (Blue - Red)
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            red_band (int): Index of the red band
            nir_band (int): Index of the near-infrared band
            blue_band (int): Index of the blue band
            gamma (float): Atmospheric resistance parameter (default: 1.0)
            
        Returns:
            np.ndarray: ARVI values typically ranging from -1 to 1
        """
        
        # Use default parameter if not provided
        gamma = gamma if gamma is not None else self.default_params['ARVI']['gamma']
        
        try:
            # Extract bands
            red = imagery[red_band].astype(np.float64)
            nir = imagery[nir_band].astype(np.float64)
            blue = imagery[blue_band].astype(np.float64)
            
            # Calculate atmospherically resistant red band
            rb = red - gamma * (blue - red)
            
            # Calculate ARVI
            numerator = nir - rb
            denominator = nir + rb
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                arvi = np.divide(numerator, denominator,
                               out=np.zeros_like(numerator),
                               where=(denominator != 0))
            
            # Set invalid values to NaN
            arvi = np.where(np.isfinite(arvi), arvi, np.nan)
            
            # Clip to reasonable range
            arvi = np.clip(arvi, -1, 1)
            
            return arvi
            
        except Exception as e:
            raise Exception(f"Error calculating ARVI: {str(e)}")
    
    def calculate_gndvi(self, imagery: np.ndarray,
                       green_band: int = 1, nir_band: int = 3) -> np.ndarray:
        """
        Calculate Green Normalized Difference Vegetation Index (GNDVI).
        
        GNDVI = (NIR - Green) / (NIR + Green)
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            green_band (int): Index of the green band (typically 525-600 nm)
            nir_band (int): Index of the near-infrared band
            
        Returns:
            np.ndarray: GNDVI values ranging from -1 to 1
        """
        
        try:
            # Extract bands
            green = imagery[green_band].astype(np.float64)
            nir = imagery[nir_band].astype(np.float64)
            
            # Calculate GNDVI
            numerator = nir - green
            denominator = nir + green
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                gndvi = np.divide(numerator, denominator,
                                out=np.zeros_like(numerator),
                                where=(denominator != 0))
            
            # Set invalid values to NaN
            gndvi = np.where(np.isfinite(gndvi), gndvi, np.nan)
            
            # Clip to valid range
            gndvi = np.clip(gndvi, -1, 1)
            
            return gndvi
            
        except Exception as e:
            raise Exception(f"Error calculating GNDVI: {str(e)}")
    
    def calculate_cvi(self, imagery: np.ndarray,
                     red_band: int = 0, nir_band: int = 3, green_band: int = 1) -> np.ndarray:
        """
        Calculate Chlorophyll Vegetation Index (CVI).
        
        CVI = (NIR / Green) * (Red / Green)
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            red_band (int): Index of the red band
            nir_band (int): Index of the near-infrared band
            green_band (int): Index of the green band
            
        Returns:
            np.ndarray: CVI values
        """
        
        try:
            # Extract bands
            red = imagery[red_band].astype(np.float64)
            nir = imagery[nir_band].astype(np.float64)
            green = imagery[green_band].astype(np.float64)
            
            # Calculate CVI
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cvi = np.divide(nir, green, out=np.zeros_like(nir), 
                              where=(green != 0)) * \
                      np.divide(red, green, out=np.zeros_like(red), 
                              where=(green != 0))
            
            # Set invalid values to NaN
            cvi = np.where(np.isfinite(cvi), cvi, np.nan)
            
            return cvi
            
        except Exception as e:
            raise Exception(f"Error calculating CVI: {str(e)}")
    
    def calculate_multiple_indices(self, imagery: np.ndarray,
                                 indices: list = None,
                                 band_config: dict = None) -> dict:
        """
        Calculate multiple vegetation indices at once.
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            indices (list): List of indices to calculate (default: ['NDVI', 'EVI', 'SAVI'])
            band_config (dict): Band configuration mapping
            
        Returns:
            dict: Dictionary containing calculated indices
        """
        
        # Default indices to calculate
        if indices is None:
            indices = ['NDVI', 'EVI', 'SAVI']
        
        # Default band configuration (assumes MODIS-like arrangement)
        if band_config is None:
            band_config = {
                'red': 0,
                'green': 1, 
                'blue': 2,
                'nir': 3
            }
        
        results = {}
        
        try:
            for index in indices:
                if index == 'NDVI':
                    results['NDVI'] = self.calculate_ndvi(
                        imagery, band_config['red'], band_config['nir']
                    )
                elif index == 'EVI':
                    results['EVI'] = self.calculate_evi(
                        imagery, band_config['red'], band_config['nir'], band_config['blue']
                    )
                elif index == 'SAVI':
                    results['SAVI'] = self.calculate_savi(
                        imagery, band_config['red'], band_config['nir']
                    )
                elif index == 'ARVI':
                    results['ARVI'] = self.calculate_arvi(
                        imagery, band_config['red'], band_config['nir'], band_config['blue']
                    )
                elif index == 'GNDVI':
                    results['GNDVI'] = self.calculate_gndvi(
                        imagery, band_config['green'], band_config['nir']
                    )
                elif index == 'CVI':
                    results['CVI'] = self.calculate_cvi(
                        imagery, band_config['red'], band_config['nir'], band_config['green']
                    )
                else:
                    print(f"Warning: Index '{index}' not recognized. Skipping.")
            
            return results
            
        except Exception as e:
            raise Exception(f"Error calculating multiple indices: {str(e)}")
    
    def detect_bloom_from_vi(self, vegetation_index: np.ndarray,
                           index_type: str = 'NDVI',
                           method: str = 'threshold') -> np.ndarray:
        """
        Detect bloom events from vegetation index data.
        
        Args:
            vegetation_index (np.ndarray): Vegetation index values
            index_type (str): Type of vegetation index ('NDVI', 'EVI', etc.)
            method (str): Detection method ('threshold', 'percentile', 'adaptive')
            
        Returns:
            np.ndarray: Binary bloom detection mask
        """
        
        try:
            # Define default thresholds for different indices
            default_thresholds = {
                'NDVI': 0.4,
                'EVI': 0.3,
                'SAVI': 0.35,
                'ARVI': 0.4,
                'GNDVI': 0.35,
                'CVI': 1.5
            }
            
            if method == 'threshold':
                threshold = default_thresholds.get(index_type, 0.4)
                bloom_mask = vegetation_index > threshold
                
            elif method == 'percentile':
                # Use 75th percentile as threshold
                valid_values = vegetation_index[~np.isnan(vegetation_index)]
                if len(valid_values) > 0:
                    threshold = np.percentile(valid_values, 75)
                    bloom_mask = vegetation_index > threshold
                else:
                    bloom_mask = np.zeros_like(vegetation_index, dtype=bool)
                    
            elif method == 'adaptive':
                # Use mean + 1.5 * standard deviation as threshold
                valid_values = vegetation_index[~np.isnan(vegetation_index)]
                if len(valid_values) > 0:
                    threshold = np.mean(valid_values) + 1.5 * np.std(valid_values)
                    bloom_mask = vegetation_index > threshold
                else:
                    bloom_mask = np.zeros_like(vegetation_index, dtype=bool)
                    
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            # Convert to binary mask
            return bloom_mask.astype(np.uint8)
            
        except Exception as e:
            raise Exception(f"Error in bloom detection: {str(e)}")
    
    def validate_imagery_bands(self, imagery: np.ndarray, 
                             required_bands: int = 4) -> bool:
        """
        Validate that imagery has sufficient bands for vegetation index calculation.
        
        Args:
            imagery (np.ndarray): Multi-band satellite imagery
            required_bands (int): Minimum number of bands required
            
        Returns:
            bool: True if imagery has sufficient bands
        """
        
        if imagery.ndim != 3:
            print("Warning: Imagery should be 3-dimensional (bands, height, width)")
            return False
        
        if imagery.shape[0] < required_bands:
            print(f"Warning: Imagery has {imagery.shape[0]} bands, but {required_bands} are required")
            return False
        
        # Check for valid reflectance values
        if np.any(imagery < 0) or np.any(imagery > 1):
            print("Warning: Some reflectance values are outside the expected range (0-1)")
        
        return True
    
    def get_index_statistics(self, vegetation_index: np.ndarray) -> dict:
        """
        Calculate comprehensive statistics for vegetation index values.
        
        Args:
            vegetation_index (np.ndarray): Vegetation index array
            
        Returns:
            dict: Dictionary containing various statistics
        """
        
        # Remove NaN values for statistics
        valid_values = vegetation_index[~np.isnan(vegetation_index)]
        
        if len(valid_values) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'percentiles': {p: np.nan for p in [10, 25, 50, 75, 90]}
            }
        
        stats = {
            'count': len(valid_values),
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values),
            'percentiles': {}
        }
        
        # Calculate percentiles
        for p in [10, 25, 50, 75, 90]:
            stats['percentiles'][p] = np.percentile(valid_values, p)
        
        return stats
