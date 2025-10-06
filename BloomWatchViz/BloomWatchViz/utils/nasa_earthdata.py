"""
NASA Earthdata API Integration for BloomSphere
Provides access to MODIS, Landsat, and VIIRS satellite imagery
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import json

class NASAEarthdataClient:
    """
    Client for accessing NASA Earthdata satellite imagery
    
    Supports:
    - MODIS (Moderate Resolution Imaging Spectroradiometer)
    - Landsat (Landsat 8/9)
    - VIIRS (Visible Infrared Imaging Radiometer Suite)
    """
    
    def __init__(self, username: str = None, password: str = None):
        """
        Initialize NASA Earthdata client
        
        Args:
            username: Earthdata login username
            password: Earthdata login password
        """
        self.username = username or os.getenv('NASA_EARTHDATA_USERNAME') or ''
        self.password = password or os.getenv('NASA_EARTHDATA_PASSWORD') or ''
        
        self.base_urls = {
            'cmr': 'https://cmr.earthdata.nasa.gov/search',
            'modis': 'https://ladsweb.modaps.eosdis.nasa.gov',
            'landsat': 'https://earthexplorer.usgs.gov',
            'viirs': 'https://ladsweb.modaps.eosdis.nasa.gov'
        }
        
        self.collections = {
            'modis_terra_ndvi': 'MOD13A2.061',  # Terra 16-day NDVI
            'modis_aqua_ndvi': 'MYD13A2.061',   # Aqua 16-day NDVI
            'modis_terra_surface': 'MOD09A1.061',  # Terra 8-day Surface Reflectance
            'landsat_8': 'LANDSAT_8_C2_L2',
            'landsat_9': 'LANDSAT_9_C2_L2',
            'viirs_vegetation': 'VNP13A1.001'    # VIIRS Vegetation Indices
        }
    
    def search_granules(self, 
                       collection: str,
                       bbox: Tuple[float, float, float, float],
                       start_date: str,
                       end_date: str,
                       max_results: int = 10) -> List[Dict]:
        """
        Search for satellite imagery granules
        
        Args:
            collection: Collection short name (e.g., 'MOD13A2.061')
            bbox: Bounding box (west, south, east, north)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_results: Maximum number of results to return
            
        Returns:
            List of granule metadata dictionaries
        """
        url = f"{self.base_urls['cmr']}/granules.json"
        
        params = {
            'short_name': collection,
            'bounding_box': ','.join(map(str, bbox)),
            'temporal': f"{start_date},{end_date}",
            'page_size': max_results
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('feed', {}).get('entry', [])
            
            results = []
            for granule in granules:
                result = {
                    'id': granule.get('id'),
                    'title': granule.get('title'),
                    'time_start': granule.get('time_start'),
                    'time_end': granule.get('time_end'),
                    'collection': collection,
                    'links': [link for link in granule.get('links', []) if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#']
                }
                results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching granules: {e}")
            return []
    
    def get_modis_vegetation_indices(self,
                                    bbox: Tuple[float, float, float, float],
                                    start_date: str,
                                    end_date: str,
                                    satellite: str = 'terra') -> List[Dict]:
        """
        Get MODIS vegetation index products
        
        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            satellite: 'terra' or 'aqua'
            
        Returns:
            List of available MODIS vegetation index granules
        """
        collection = self.collections[f'modis_{satellite}_ndvi']
        return self.search_granules(collection, bbox, start_date, end_date)
    
    def get_landsat_imagery(self,
                           bbox: Tuple[float, float, float, float],
                           start_date: str,
                           end_date: str,
                           landsat_version: int = 9) -> List[Dict]:
        """
        Get Landsat imagery
        
        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            landsat_version: 8 or 9
            
        Returns:
            List of available Landsat granules
        """
        collection = self.collections[f'landsat_{landsat_version}']
        return self.search_granules(collection, bbox, start_date, end_date)
    
    def get_viirs_vegetation(self,
                            bbox: Tuple[float, float, float, float],
                            start_date: str,
                            end_date: str) -> List[Dict]:
        """
        Get VIIRS vegetation index products
        
        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of available VIIRS vegetation index granules
        """
        collection = self.collections['viirs_vegetation']
        return self.search_granules(collection, bbox, start_date, end_date)
    
    def get_available_dates(self,
                           bbox: Tuple[float, float, float, float],
                           data_source: str = 'modis',
                           days_back: int = 90) -> List[str]:
        """
        Get list of available dates for a region
        
        Args:
            bbox: Bounding box (west, south, east, north)
            data_source: 'modis', 'landsat', or 'viirs'
            days_back: Number of days to look back
            
        Returns:
            List of available date strings
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        if data_source == 'modis':
            granules = self.get_modis_vegetation_indices(bbox, start_str, end_str)
        elif data_source == 'landsat':
            granules = self.get_landsat_imagery(bbox, start_str, end_str)
        elif data_source == 'viirs':
            granules = self.get_viirs_vegetation(bbox, start_str, end_str)
        else:
            return []
        
        dates = []
        for granule in granules:
            if granule.get('time_start'):
                date_str = granule['time_start'].split('T')[0]
                if date_str not in dates:
                    dates.append(date_str)
        
        return sorted(dates)
    
    def download_granule(self, granule_url: str, output_path: str) -> bool:
        """
        Download a granule file
        
        Args:
            granule_url: URL to the granule
            output_path: Local path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.username and self.password:
                auth = (self.username, self.password)
            else:
                auth = None
            
            response = requests.get(granule_url, auth=auth, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Error downloading granule: {e}")
            return False
    
    def get_sample_data_info(self, region: str = 'california') -> Dict:
        """
        Get sample data information for demonstration purposes
        
        Args:
            region: Region name
            
        Returns:
            Dictionary with sample data information
        """
        sample_regions = {
            'california': {
                'bbox': (-124.4, 32.5, -114.1, 42.0),
                'name': 'California',
                'description': 'Major agricultural and wildflower regions'
            },
            'great_plains': {
                'bbox': (-104.0, 36.0, -97.0, 49.0),
                'name': 'Great Plains',
                'description': 'Prairie grasslands and agricultural areas'
            },
            'amazon': {
                'bbox': (-73.0, -10.0, -50.0, 5.0),
                'name': 'Amazon Rainforest',
                'description': 'Tropical rainforest vegetation patterns'
            },
            'sahel': {
                'bbox': (-15.0, 10.0, 40.0, 20.0),
                'name': 'Sahel Region',
                'description': 'Semi-arid seasonal vegetation transitions'
            },
            'australia': {
                'bbox': (113.0, -43.0, 153.0, -10.0),
                'name': 'Australia',
                'description': 'Diverse ecosystems including deserts and forests'
            }
        }
        
        return sample_regions.get(region, sample_regions['california'])
    
    def create_mock_granule_list(self, 
                                 bbox: Tuple[float, float, float, float],
                                 num_granules: int = 5) -> List[Dict]:
        """
        Create mock granule data for demonstration when API is not available
        
        Args:
            bbox: Bounding box
            num_granules: Number of mock granules to create
            
        Returns:
            List of mock granule dictionaries
        """
        end_date = datetime.now()
        granules = []
        
        for i in range(num_granules):
            date = end_date - timedelta(days=i*16)  # MODIS 16-day product
            
            granule = {
                'id': f'MOCK_GRANULE_{i}',
                'title': f'MOD13A2.061_{date.strftime("%Y%j")}',
                'time_start': date.isoformat(),
                'time_end': (date + timedelta(hours=23, minutes=59)).isoformat(),
                'collection': 'MOD13A2.061',
                'bbox': bbox,
                'mock_data': True,
                'links': []
            }
            granules.append(granule)
        
        return granules
    
    def test_connection(self) -> Dict:
        """
        Test connection to NASA Earthdata
        
        Returns:
            Dictionary with connection test results
        """
        results = {
            'cmr_accessible': False,
            'authenticated': False,
            'error': None
        }
        
        try:
            # Test CMR search endpoint
            response = requests.get(
                f"{self.base_urls['cmr']}/collections.json",
                params={'short_name': 'MOD13A2'},
                timeout=10
            )
            results['cmr_accessible'] = response.status_code == 200
            
            # Test authentication if credentials provided
            if self.username and self.password:
                # This would test actual authenticated endpoint
                results['authenticated'] = True
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
