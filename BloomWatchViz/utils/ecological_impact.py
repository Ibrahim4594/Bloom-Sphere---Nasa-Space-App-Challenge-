"""
Ecological Impact Analyzer for BloomSphere
Provides insights on pollinator activity, crop yield, and pollen forecasts
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class EcologicalImpactAnalyzer:
    """
    Analyzes ecological impacts of bloom events
    
    Provides insights on:
    - Pollinator activity predictions
    - Crop yield forecasts
    - Pollen production estimates
    - Ecosystem services valuation
    """
    
    def __init__(self):
        """Initialize impact analyzer"""
        
        # Pollinator activity parameters by bloom type
        self.pollinator_factors = {
            'wildflowers': 0.95,
            'orchard': 0.90,
            'agricultural': 0.70,
            'forest': 0.60,
            'grassland': 0.75,
            'default': 0.65
        }
        
        # Pollen production levels by vegetation type
        self.pollen_levels = {
            'grass': 'Very High',
            'trees': 'High',
            'wildflowers': 'Moderate',
            'agricultural': 'Low',
            'desert_bloom': 'Moderate',
            'default': 'Moderate'
        }
    
    def predict_pollinator_activity(self,
                                    bloom_intensity: float,
                                    vegetation_type: str,
                                    temperature: float = 20.0,
                                    day_of_year: int = 120) -> Dict:
        """
        Predict pollinator activity based on bloom conditions
        
        Args:
            bloom_intensity: NDVI or bloom coverage (0-1)
            vegetation_type: Type of vegetation
            temperature: Average temperature in Celsius
            day_of_year: Day of year for seasonal adjustment
            
        Returns:
            Dictionary with pollinator activity prediction
        """
        # Base activity from bloom intensity
        base_activity = bloom_intensity * 100
        
        # Vegetation type factor
        veg_factor = self.pollinator_factors.get(
            vegetation_type.lower(),
            self.pollinator_factors['default']
        )
        
        # Temperature factor (optimal 15-25°C)
        if 15 <= temperature <= 25:
            temp_factor = 1.0
        elif temperature < 15:
            temp_factor = max(0.3, (temperature - 5) / 10)
        else:  # temperature > 25
            temp_factor = max(0.5, 1 - (temperature - 25) / 20)
        
        # Seasonal factor (peak spring-summer)
        peak_days = [90, 180]  # April-June
        seasonal_factor = 1.0
        for peak in peak_days:
            distance = min(abs(day_of_year - peak), 365 - abs(day_of_year - peak))
            if distance < 60:
                seasonal_factor = max(seasonal_factor, 1.0 - distance / 120)
        
        # Calculate overall activity level
        activity_score = base_activity * veg_factor * temp_factor * seasonal_factor
        activity_score = np.clip(activity_score, 0, 100)
        
        # Categorize activity
        if activity_score >= 75:
            level = 'Very High'
            description = 'Excellent conditions for pollinators. Peak activity expected.'
        elif activity_score >= 50:
            level = 'High'
            description = 'Good pollinator activity. Favorable bloom conditions.'
        elif activity_score >= 30:
            level = 'Moderate'
            description = 'Moderate pollinator presence. Adequate bloom support.'
        else:
            level = 'Low'
            description = 'Limited pollinator activity. Suboptimal conditions.'
        
        # Estimate bee colony foraging range impact
        foraging_radius_km = 3.0  # Average bee foraging radius
        impact_area_km2 = np.pi * foraging_radius_km ** 2
        
        return {
            'activity_score': float(activity_score),
            'activity_level': level,
            'description': description,
            'vegetation_factor': float(veg_factor),
            'temperature_factor': float(temp_factor),
            'seasonal_factor': float(seasonal_factor),
            'estimated_foraging_area_km2': float(impact_area_km2),
            'dominant_pollinators': self._identify_pollinators(vegetation_type),
            'recommendations': self._get_pollinator_recommendations(activity_score, vegetation_type)
        }
    
    def _identify_pollinators(self, vegetation_type: str) -> List[str]:
        """Identify likely pollinator species for vegetation type"""
        pollinator_map = {
            'wildflowers': ['Honeybees', 'Bumblebees', 'Butterflies', 'Native bees'],
            'orchard': ['Honeybees', 'Bumblebees', 'Mason bees'],
            'agricultural': ['Honeybees', 'Native bees'],
            'forest': ['Bumblebees', 'Butterflies', 'Moths'],
            'grassland': ['Native bees', 'Butterflies', 'Beetles'],
            'desert_bloom': ['Native bees', 'Hummingbirds', 'Moths']
        }
        
        return pollinator_map.get(vegetation_type.lower(), ['Honeybees', 'Native pollinators'])
    
    def _get_pollinator_recommendations(self, activity_score: float, veg_type: str) -> List[str]:
        """Get recommendations for supporting pollinators"""
        recommendations = []
        
        if activity_score > 60:
            recommendations.append("Excellent time for observing pollinator activity")
            recommendations.append("Consider beekeeping operations during peak bloom")
        
        recommendations.append("Maintain diverse flowering plants for continuous bloom")
        recommendations.append("Minimize pesticide use during bloom period")
        
        if veg_type.lower() in ['agricultural', 'orchard']:
            recommendations.append("Optimal window for crop pollination services")
        
        return recommendations
    
    def forecast_crop_yield(self,
                           bloom_coverage: float,
                           bloom_intensity: float,
                           crop_type: str,
                           pollinator_activity: float) -> Dict:
        """
        Forecast crop yield based on bloom characteristics
        
        Args:
            bloom_coverage: Percentage of area in bloom (0-100)
            bloom_intensity: Average bloom NDVI
            crop_type: Type of crop
            pollinator_activity: Pollinator activity score (0-100)
            
        Returns:
            Dictionary with yield forecast
        """
        # Base yield index from bloom characteristics
        base_yield = (bloom_coverage / 100) * bloom_intensity * 100
        
        # Crop-specific pollinator dependency
        pollinator_dependency = {
            'orchard': 0.80,  # Highly dependent
            'berry': 0.85,
            'vegetables': 0.60,
            'wheat': 0.10,  # Wind pollinated
            'corn': 0.05,
            'soybean': 0.40,
            'default': 0.50
        }
        
        dependency = pollinator_dependency.get(crop_type.lower(), pollinator_dependency['default'])
        
        # Pollinator contribution to yield
        pollinator_contribution = (pollinator_activity / 100) * dependency * 30
        
        # Calculate yield index
        yield_index = base_yield + pollinator_contribution
        yield_index = np.clip(yield_index, 0, 100)
        
        # Yield outlook
        if yield_index >= 80:
            outlook = 'Excellent'
            expected_yield = 'Above average (110-120%)'
        elif yield_index >= 65:
            outlook = 'Good'
            expected_yield = 'Above average (100-110%)'
        elif yield_index >= 50:
            outlook = 'Fair'
            expected_yield = 'Average (90-100%)'
        elif yield_index >= 35:
            outlook = 'Below Average'
            expected_yield = 'Below average (75-90%)'
        else:
            outlook = 'Poor'
            expected_yield = 'Significantly below average (<75%)'
        
        return {
            'yield_index': float(yield_index),
            'outlook': outlook,
            'expected_yield_pct': expected_yield,
            'bloom_contribution': float(base_yield),
            'pollinator_contribution': float(pollinator_contribution),
            'pollinator_dependency': float(dependency),
            'recommendations': self._get_yield_recommendations(yield_index, crop_type)
        }
    
    def _get_yield_recommendations(self, yield_index: float, crop_type: str) -> List[str]:
        """Get crop management recommendations based on yield forecast"""
        recommendations = []
        
        if yield_index < 50:
            recommendations.append("Consider supplemental irrigation if available")
            recommendations.append("Monitor for pest and disease pressure")
            recommendations.append("Evaluate nutrient management practices")
        
        if crop_type.lower() in ['orchard', 'berry', 'vegetables']:
            recommendations.append("Ensure adequate pollinator habitat nearby")
            recommendations.append("Consider managed pollinator introduction")
        
        if yield_index > 65:
            recommendations.append("Plan for increased harvest capacity")
            recommendations.append("Optimize post-harvest handling")
        
        return recommendations
    
    def estimate_pollen_production(self,
                                   vegetation_type: str,
                                   bloom_coverage_km2: float,
                                   bloom_intensity: float,
                                   day_of_year: int = 120) -> Dict:
        """
        Estimate pollen production and allergy risk
        
        Args:
            vegetation_type: Type of vegetation
            bloom_coverage_km2: Area covered by bloom in km²
            bloom_intensity: Bloom NDVI intensity
            day_of_year: Day of year
            
        Returns:
            Dictionary with pollen forecast
        """
        # Pollen production rate (arbitrary units per km²)
        production_rates = {
            'grass': 1000,
            'trees': 800,
            'agricultural': 300,
            'wildflowers': 400,
            'orchard': 600,
            'default': 500
        }
        
        production_rate = production_rates.get(
            vegetation_type.lower(),
            production_rates['default']
        )
        
        # Calculate total pollen production
        total_production = bloom_coverage_km2 * bloom_intensity * production_rate
        
        # Allergy risk level
        pollen_level = self.pollen_levels.get(
            vegetation_type.lower(),
            self.pollen_levels['default']
        )
        
        # Risk score (0-100)
        risk_score = min(100, (total_production / 1000) * 10)
        
        if risk_score >= 75:
            risk_level = 'Very High'
            advisory = 'Extreme allergy risk. Sensitive individuals should stay indoors.'
        elif risk_score >= 50:
            risk_level = 'High'
            advisory = 'High allergy risk. Take preventive measures if sensitive.'
        elif risk_score >= 30:
            risk_level = 'Moderate'
            advisory = 'Moderate allergy risk. Monitor symptoms.'
        else:
            risk_level = 'Low'
            advisory = 'Low allergy risk. Generally safe for most people.'
        
        # Peak pollen times
        peak_times = self._get_peak_pollen_times(day_of_year)
        
        return {
            'total_pollen_production': float(total_production),
            'pollen_level': pollen_level,
            'allergy_risk_score': float(risk_score),
            'risk_level': risk_level,
            'advisory': advisory,
            'peak_pollen_times': peak_times,
            'affected_area_km2': float(bloom_coverage_km2),
            'recommendations': self._get_pollen_recommendations(risk_level)
        }
    
    def _get_peak_pollen_times(self, day_of_year: int) -> Dict:
        """Get peak pollen release times"""
        # Pollen typically peaks in morning
        return {
            'daily_peak': '6 AM - 10 AM',
            'season': 'Spring' if 60 < day_of_year < 150 else 
                     'Summer' if 150 <= day_of_year < 240 else
                     'Fall' if 240 <= day_of_year < 330 else 'Winter',
            'weather_dependency': 'Higher on warm, dry, windy days'
        }
    
    def _get_pollen_recommendations(self, risk_level: str) -> List[str]:
        """Get recommendations for pollen management"""
        recommendations = []
        
        if risk_level in ['High', 'Very High']:
            recommendations.append("Keep windows closed during peak hours")
            recommendations.append("Use air filtration systems indoors")
            recommendations.append("Shower after being outdoors")
            recommendations.append("Take allergy medication as prescribed")
        else:
            recommendations.append("Monitor pollen forecasts daily")
            recommendations.append("Be prepared with allergy relief if sensitive")
        
        return recommendations
    
    def assess_ecosystem_services(self,
                                 bloom_area_km2: float,
                                 vegetation_type: str,
                                 bloom_health: float) -> Dict:
        """
        Assess economic value of ecosystem services
        
        Args:
            bloom_area_km2: Area of bloom in km²
            vegetation_type: Type of vegetation
            bloom_health: Bloom intensity (0-1)
            
        Returns:
            Dictionary with ecosystem services valuation
        """
        # Service values per km² per year (USD)
        service_values = {
            'pollination': 1000,
            'carbon_sequestration': 500,
            'water_regulation': 300,
            'soil_formation': 200,
            'air_quality': 400,
            'biodiversity': 600,
            'recreation': 800
        }
        
        # Vegetation type multipliers
        type_multipliers = {
            'forest': 1.5,
            'wetland': 1.8,
            'grassland': 1.0,
            'agricultural': 0.7,
            'wildflowers': 1.2,
            'default': 1.0
        }
        
        multiplier = type_multipliers.get(
            vegetation_type.lower(),
            type_multipliers['default']
        )
        
        # Calculate service values
        services = {}
        total_value = 0
        
        for service, base_value in service_values.items():
            annual_value = bloom_area_km2 * base_value * multiplier * bloom_health
            services[service] = {
                'annual_value_usd': float(annual_value),
                'per_hectare_usd': float(annual_value / (bloom_area_km2 * 100))
            }
            total_value += annual_value
        
        return {
            'total_annual_value_usd': float(total_value),
            'per_km2_value_usd': float(total_value / bloom_area_km2) if bloom_area_km2 > 0 else 0,
            'services_breakdown': services,
            'vegetation_type': vegetation_type,
            'health_factor': float(bloom_health),
            'area_km2': float(bloom_area_km2),
            'key_services': self._identify_key_services(vegetation_type)
        }
    
    def _identify_key_services(self, vegetation_type: str) -> List[str]:
        """Identify key ecosystem services by vegetation type"""
        service_map = {
            'forest': ['Carbon Sequestration', 'Biodiversity', 'Water Regulation'],
            'wetland': ['Water Purification', 'Flood Control', 'Biodiversity'],
            'grassland': ['Pollination', 'Soil Formation', 'Carbon Storage'],
            'agricultural': ['Food Production', 'Pollination Services'],
            'wildflowers': ['Pollination', 'Biodiversity', 'Recreation'],
            'default': ['Pollination', 'Carbon Sequestration', 'Biodiversity']
        }
        
        return service_map.get(vegetation_type.lower(), service_map['default'])
    
    def generate_impact_report(self,
                              bloom_data: Dict,
                              region_name: str,
                              date: Optional[datetime] = None) -> Dict:
        """
        Generate comprehensive ecological impact report
        
        Args:
            bloom_data: Dictionary with bloom characteristics
            region_name: Name of the region
            date: Date of analysis
            
        Returns:
            Comprehensive impact report
        """
        if date is None:
            date = datetime.now()
        
        # Extract bloom parameters
        coverage = bloom_data.get('coverage_percent', 50)
        intensity = bloom_data.get('intensity', 0.6)
        area_km2 = bloom_data.get('area_km2', 100)
        veg_type = bloom_data.get('vegetation_type', 'wildflowers')
        
        # Calculate all impacts
        pollinator = self.predict_pollinator_activity(
            intensity, veg_type, day_of_year=date.timetuple().tm_yday
        )
        
        crop_yield = self.forecast_crop_yield(
            coverage, intensity, veg_type, pollinator['activity_score']
        )
        
        pollen = self.estimate_pollen_production(
            veg_type, area_km2, intensity, date.timetuple().tm_yday
        )
        
        ecosystem = self.assess_ecosystem_services(
            area_km2, veg_type, intensity
        )
        
        report = {
            'report_date': date.isoformat(),
            'region': region_name,
            'bloom_summary': {
                'coverage_percent': coverage,
                'intensity': intensity,
                'area_km2': area_km2,
                'vegetation_type': veg_type
            },
            'pollinator_activity': pollinator,
            'crop_yield_forecast': crop_yield,
            'pollen_forecast': pollen,
            'ecosystem_services': ecosystem,
            'overall_assessment': self._generate_overall_assessment(
                pollinator, crop_yield, pollen, ecosystem
            )
        }
        
        return report
    
    def _generate_overall_assessment(self, pollinator, crop_yield, pollen, ecosystem) -> str:
        """Generate overall impact assessment summary"""
        assessments = []
        
        if pollinator['activity_score'] > 70:
            assessments.append("Excellent pollinator conditions")
        
        if crop_yield['yield_index'] > 65:
            assessments.append("Favorable crop yield prospects")
        
        if pollen['risk_level'] in ['High', 'Very High']:
            assessments.append("Elevated pollen allergy risk")
        
        if ecosystem['total_annual_value_usd'] > 1000000:
            assessments.append("Significant ecosystem service value")
        
        if not assessments:
            return "Moderate ecological impacts expected"
        
        return "; ".join(assessments) + "."
