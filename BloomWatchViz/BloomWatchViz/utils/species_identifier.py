"""
Species Identification Module for BloomSphere
Suggests plant species based on spectral signatures and vegetation indices
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class SpeciesIdentifier:
    """
    Identifies plant species/types based on spectral signatures
    
    Uses vegetation indices (NDVI, EVI, SAVI) and spectral characteristics
    to suggest likely plant species or vegetation types
    """
    
    def __init__(self):
        """Initialize species identifier with reference signatures"""
        
        # Reference spectral signatures for common vegetation types
        # Format: (NDVI_range, EVI_range, red_reflectance, NIR_reflectance)
        self.species_database = {
            # Agricultural Crops
            'corn': {
                'ndvi_range': (0.6, 0.9),
                'evi_range': (0.5, 0.8),
                'red_reflectance': (0.05, 0.15),
                'nir_reflectance': (0.35, 0.50),
                'category': 'Agricultural',
                'bloom_season': 'Summer',
                'description': 'Maize/Corn - Row crop with high biomass'
            },
            'wheat': {
                'ndvi_range': (0.5, 0.8),
                'evi_range': (0.4, 0.7),
                'red_reflectance': (0.08, 0.18),
                'nir_reflectance': (0.30, 0.45),
                'category': 'Agricultural',
                'bloom_season': 'Spring',
                'description': 'Wheat - Cereal grain crop'
            },
            'soybean': {
                'ndvi_range': (0.6, 0.85),
                'evi_range': (0.5, 0.75),
                'red_reflectance': (0.06, 0.16),
                'nir_reflectance': (0.32, 0.48),
                'category': 'Agricultural',
                'bloom_season': 'Summer',
                'description': 'Soybean - Legume crop'
            },
            
            # Forest Types
            'deciduous_forest': {
                'ndvi_range': (0.4, 0.8),
                'evi_range': (0.3, 0.6),
                'red_reflectance': (0.04, 0.12),
                'nir_reflectance': (0.28, 0.45),
                'category': 'Forest',
                'bloom_season': 'Spring',
                'description': 'Deciduous Forest - Broadleaf trees that shed leaves'
            },
            'evergreen_forest': {
                'ndvi_range': (0.5, 0.75),
                'evi_range': (0.35, 0.6),
                'red_reflectance': (0.03, 0.10),
                'nir_reflectance': (0.30, 0.42),
                'category': 'Forest',
                'bloom_season': 'Year-round',
                'description': 'Evergreen Forest - Coniferous trees'
            },
            
            # Grasslands and Prairies
            'grassland': {
                'ndvi_range': (0.3, 0.6),
                'evi_range': (0.2, 0.5),
                'red_reflectance': (0.10, 0.20),
                'nir_reflectance': (0.25, 0.40),
                'category': 'Grassland',
                'bloom_season': 'Spring-Summer',
                'description': 'Grassland - Mixed grasses and herbs'
            },
            'prairie': {
                'ndvi_range': (0.35, 0.65),
                'evi_range': (0.25, 0.55),
                'red_reflectance': (0.08, 0.18),
                'nir_reflectance': (0.26, 0.42),
                'category': 'Grassland',
                'bloom_season': 'Spring',
                'description': 'Prairie - Native grassland ecosystem'
            },
            
            # Flowering Plants
            'wildflowers': {
                'ndvi_range': (0.4, 0.75),
                'evi_range': (0.3, 0.65),
                'red_reflectance': (0.06, 0.16),
                'nir_reflectance': (0.28, 0.45),
                'category': 'Flowering',
                'bloom_season': 'Spring-Summer',
                'description': 'Wildflowers - Mixed flowering species'
            },
            'desert_bloom': {
                'ndvi_range': (0.2, 0.5),
                'evi_range': (0.15, 0.4),
                'red_reflectance': (0.15, 0.30),
                'nir_reflectance': (0.20, 0.35),
                'category': 'Desert',
                'bloom_season': 'Spring',
                'description': 'Desert Bloom - Ephemeral flowering after rain'
            },
            
            # Orchards and Vineyards
            'orchard': {
                'ndvi_range': (0.5, 0.8),
                'evi_range': (0.4, 0.7),
                'red_reflectance': (0.05, 0.15),
                'nir_reflectance': (0.30, 0.47),
                'category': 'Agricultural',
                'bloom_season': 'Spring',
                'description': 'Fruit Orchard - Apple, cherry, peach trees'
            },
            'vineyard': {
                'ndvi_range': (0.4, 0.7),
                'evi_range': (0.3, 0.6),
                'red_reflectance': (0.07, 0.17),
                'nir_reflectance': (0.28, 0.44),
                'category': 'Agricultural',
                'bloom_season': 'Spring',
                'description': 'Vineyard - Grapevines'
            },
            
            # Shrublands
            'shrubland': {
                'ndvi_range': (0.3, 0.6),
                'evi_range': (0.2, 0.5),
                'red_reflectance': (0.10, 0.22),
                'nir_reflectance': (0.24, 0.40),
                'category': 'Shrubland',
                'bloom_season': 'Spring-Summer',
                'description': 'Shrubland - Woody shrubs and bushes'
            },
            
            # Wetlands
            'wetland': {
                'ndvi_range': (0.4, 0.7),
                'evi_range': (0.3, 0.6),
                'red_reflectance': (0.06, 0.16),
                'nir_reflectance': (0.26, 0.43),
                'category': 'Wetland',
                'bloom_season': 'Spring-Summer',
                'description': 'Wetland - Marsh and aquatic vegetation'
            }
        }
    
    def identify_species(self,
                        ndvi: float,
                        evi: float,
                        red: Optional[float] = None,
                        nir: Optional[float] = None,
                        top_n: int = 3) -> List[Dict]:
        """
        Identify most likely species based on spectral characteristics
        
        Args:
            ndvi: NDVI value
            evi: EVI value
            red: Red band reflectance (optional)
            nir: Near-infrared reflectance (optional)
            top_n: Number of top matches to return
            
        Returns:
            List of species matches with confidence scores
        """
        matches = []
        
        for species_name, characteristics in self.species_database.items():
            # Calculate match score
            score = 0.0
            max_score = 0.0
            
            # NDVI matching (weighted 40%)
            ndvi_min, ndvi_max = characteristics['ndvi_range']
            if ndvi_min <= ndvi <= ndvi_max:
                ndvi_score = 1.0
            else:
                # Penalty for being outside range
                if ndvi < ndvi_min:
                    ndvi_score = max(0, 1 - (ndvi_min - ndvi) / 0.2)
                else:
                    ndvi_score = max(0, 1 - (ndvi - ndvi_max) / 0.2)
            
            score += ndvi_score * 0.4
            max_score += 0.4
            
            # EVI matching (weighted 40%)
            evi_min, evi_max = characteristics['evi_range']
            if evi_min <= evi <= evi_max:
                evi_score = 1.0
            else:
                if evi < evi_min:
                    evi_score = max(0, 1 - (evi_min - evi) / 0.2)
                else:
                    evi_score = max(0, 1 - (evi - evi_max) / 0.2)
            
            score += evi_score * 0.4
            max_score += 0.4
            
            # Red band matching (weighted 10% if available)
            if red is not None:
                red_min, red_max = characteristics['red_reflectance']
                if red_min <= red <= red_max:
                    red_score = 1.0
                else:
                    if red < red_min:
                        red_score = max(0, 1 - (red_min - red) / 0.1)
                    else:
                        red_score = max(0, 1 - (red - red_max) / 0.1)
                
                score += red_score * 0.1
                max_score += 0.1
            
            # NIR band matching (weighted 10% if available)
            if nir is not None:
                nir_min, nir_max = characteristics['nir_reflectance']
                if nir_min <= nir <= nir_max:
                    nir_score = 1.0
                else:
                    if nir < nir_min:
                        nir_score = max(0, 1 - (nir_min - nir) / 0.1)
                    else:
                        nir_score = max(0, 1 - (nir - nir_max) / 0.1)
                
                score += nir_score * 0.1
                max_score += 0.1
            
            # Normalize score
            confidence = (score / max_score) * 100 if max_score > 0 else 0
            
            matches.append({
                'species': species_name,
                'confidence': confidence,
                'category': characteristics['category'],
                'bloom_season': characteristics['bloom_season'],
                'description': characteristics['description'],
                'ndvi_range': characteristics['ndvi_range'],
                'evi_range': characteristics['evi_range']
            })
        
        # Sort by confidence and return top N
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:top_n]
    
    def categorize_vegetation(self, ndvi: float, evi: float) -> Dict:
        """
        Categorize vegetation into broad types
        
        Args:
            ndvi: NDVI value
            evi: EVI value
            
        Returns:
            Dictionary with vegetation category and characteristics
        """
        # Determine general vegetation health
        if ndvi < 0.2:
            health = 'Sparse/Stressed'
            vigor = 'Low'
        elif ndvi < 0.4:
            health = 'Moderate'
            vigor = 'Medium'
        elif ndvi < 0.6:
            health = 'Good'
            vigor = 'High'
        else:
            health = 'Excellent'
            vigor = 'Very High'
        
        # Determine vegetation density
        if ndvi < 0.3:
            density = 'Low'
        elif ndvi < 0.6:
            density = 'Medium'
        else:
            density = 'High'
        
        # Determine photosynthetic activity
        if evi < 0.3:
            activity = 'Low'
        elif evi < 0.6:
            activity = 'Moderate'
        else:
            activity = 'High'
        
        return {
            'health_status': health,
            'vigor': vigor,
            'density': density,
            'photosynthetic_activity': activity,
            'ndvi': ndvi,
            'evi': evi
        }
    
    def suggest_bloom_timing(self, species_name: str) -> Dict:
        """
        Get typical bloom timing for identified species
        
        Args:
            species_name: Name of the species
            
        Returns:
            Dictionary with bloom timing information
        """
        if species_name in self.species_database:
            characteristics = self.species_database[species_name]
            
            # Map bloom season to months
            season_to_months = {
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11],
                'Winter': [12, 1, 2],
                'Spring-Summer': [3, 4, 5, 6, 7, 8],
                'Year-round': list(range(1, 13))
            }
            
            bloom_months = season_to_months.get(
                characteristics['bloom_season'],
                []
            )
            
            return {
                'season': characteristics['bloom_season'],
                'months': bloom_months,
                'category': characteristics['category'],
                'description': characteristics['description']
            }
        
        return {
            'season': 'Unknown',
            'months': [],
            'category': 'Unknown',
            'description': 'Species not in database'
        }
    
    def analyze_spectral_signature(self, 
                                   red: float,
                                   green: float,
                                   blue: float,
                                   nir: float) -> Dict:
        """
        Comprehensive spectral signature analysis
        
        Args:
            red: Red band reflectance
            green: Green band reflectance
            blue: Blue band reflectance
            nir: Near-infrared reflectance
            
        Returns:
            Dictionary with detailed spectral analysis
        """
        # Calculate vegetation indices
        ndvi = (nir - red) / (nir + red + 1e-10)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        savi = 1.5 * ((nir - red) / (nir + red + 0.5))
        
        # Calculate additional indices
        green_ndvi = (nir - green) / (nir + green + 1e-10)
        red_edge_position = self._estimate_red_edge(red, nir)
        
        # Chlorophyll estimation (empirical relationship)
        chlorophyll_index = (nir / red) - 1
        
        # Water content estimation
        water_index = (green - nir) / (green + nir + 1e-10)
        
        analysis = {
            'vegetation_indices': {
                'NDVI': float(np.clip(ndvi, -1, 1)),
                'EVI': float(np.clip(evi, -1, 1)),
                'SAVI': float(np.clip(savi, -1, 1)),
                'Green_NDVI': float(np.clip(green_ndvi, -1, 1))
            },
            'spectral_properties': {
                'red_edge_position': red_edge_position,
                'chlorophyll_index': float(chlorophyll_index),
                'water_content_index': float(np.clip(water_index, -1, 1)),
                'nir_red_ratio': float(nir / (red + 1e-10))
            },
            'band_reflectances': {
                'red': float(red),
                'green': float(green),
                'blue': float(blue),
                'nir': float(nir)
            }
        }
        
        return analysis
    
    def _estimate_red_edge(self, red: float, nir: float) -> float:
        """
        Estimate red-edge position (simplified)
        Red edge is the steep slope in reflectance between red and NIR
        """
        # Simplified estimation - actual red edge requires more bands
        if nir > red:
            # Normalized position estimate
            red_edge = 700 + 40 * ((nir - red) / (nir + red + 1e-10))
        else:
            red_edge = 700
        
        return float(red_edge)
    
    def get_all_species_info(self) -> List[Dict]:
        """
        Get information about all species in the database
        
        Returns:
            List of all species with their characteristics
        """
        species_list = []
        
        for name, characteristics in self.species_database.items():
            species_list.append({
                'name': name,
                **characteristics
            })
        
        return species_list
    
    def get_species_by_category(self, category: str) -> List[str]:
        """
        Get all species in a specific category
        
        Args:
            category: Category name (e.g., 'Agricultural', 'Forest')
            
        Returns:
            List of species names in that category
        """
        return [
            name for name, chars in self.species_database.items()
            if chars['category'] == category
        ]
