"""
Bloom Prediction Module for BloomSphere
Uses historical vegetation index patterns to predict future bloom events
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class BloomPredictor:
    """
    Predicts bloom events based on historical vegetation index patterns
    
    Uses time-series analysis and trend detection to forecast bloom timing,
    intensity, and duration based on historical NDVI/EVI patterns
    """
    
    def __init__(self):
        """Initialize bloom predictor"""
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.fitted = False
    
    def detect_bloom_cycles(self, 
                           dates: List[datetime], 
                           ndvi_values: List[float],
                           min_cycle_days: int = 90) -> List[Dict]:
        """
        Detect periodic bloom cycles in historical data
        
        Args:
            dates: List of observation dates
            ndvi_values: Corresponding NDVI values
            min_cycle_days: Minimum days between bloom events
            
        Returns:
            List of detected bloom cycles with timing and characteristics
        """
        if len(dates) < 10:
            return []
        
        # Convert to numpy arrays
        dates_arr = np.array([d.timestamp() for d in dates])
        ndvi_arr = np.array(ndvi_values)
        
        # Normalize timestamps to days
        days = (dates_arr - dates_arr.min()) / (24 * 3600)
        
        # Find peaks (potential bloom events)
        peaks, properties = signal.find_peaks(
            ndvi_arr,
            height=0.5,  # Minimum NDVI for bloom
            distance=min_cycle_days / np.mean(np.diff(days)) if len(days) > 1 else 10,
            prominence=0.15  # Minimum prominence
        )
        
        cycles = []
        for i, peak_idx in enumerate(peaks):
            peak_date = dates[peak_idx]
            peak_ndvi = ndvi_arr[peak_idx]
            
            # Find bloom start (when NDVI starts rising)
            start_idx = peak_idx
            for j in range(peak_idx - 1, -1, -1):
                if ndvi_arr[j] < peak_ndvi * 0.6:
                    start_idx = j
                    break
            
            # Find bloom end (when NDVI drops)
            end_idx = peak_idx
            for j in range(peak_idx + 1, len(ndvi_arr)):
                if ndvi_arr[j] < peak_ndvi * 0.6:
                    end_idx = j
                    break
            
            cycle = {
                'cycle_number': i + 1,
                'peak_date': peak_date,
                'start_date': dates[start_idx],
                'end_date': dates[end_idx],
                'peak_ndvi': float(peak_ndvi),
                'duration_days': (dates[end_idx] - dates[start_idx]).days,
                'day_of_year': peak_date.timetuple().tm_yday,
                'month': peak_date.month
            }
            cycles.append(cycle)
        
        return cycles
    
    def calculate_seasonal_pattern(self, cycles: List[Dict]) -> Dict:
        """
        Calculate seasonal bloom patterns from detected cycles
        
        Args:
            cycles: List of detected bloom cycles
            
        Returns:
            Dictionary with seasonal pattern statistics
        """
        if not cycles:
            return {
                'has_pattern': False,
                'message': 'Insufficient data for pattern detection'
            }
        
        # Extract seasonal information
        peak_days = [c['day_of_year'] for c in cycles]
        peak_ndvi = [c['peak_ndvi'] for c in cycles]
        durations = [c['duration_days'] for c in cycles]
        
        # Calculate statistics
        pattern = {
            'has_pattern': True,
            'num_cycles': len(cycles),
            'avg_peak_day': int(np.mean(peak_days)),
            'std_peak_day': float(np.std(peak_days)),
            'avg_peak_ndvi': float(np.mean(peak_ndvi)),
            'std_peak_ndvi': float(np.std(peak_ndvi)),
            'avg_duration_days': int(np.mean(durations)),
            'std_duration_days': float(np.std(durations)),
            'consistency_score': self._calculate_consistency(peak_days)
        }
        
        # Determine bloom season
        avg_month = int(np.mean([c['month'] for c in cycles]))
        seasons = {
            (3, 4, 5): 'Spring',
            (6, 7, 8): 'Summer',
            (9, 10, 11): 'Fall',
            (12, 1, 2): 'Winter'
        }
        
        for months, season in seasons.items():
            if avg_month in months:
                pattern['typical_season'] = season
                break
        
        return pattern
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """
        Calculate consistency score (0-1) based on variance
        Lower variance = higher consistency
        """
        if len(values) < 2:
            return 0.5
        
        std = np.std(values)
        mean = np.mean(values)
        
        if mean == 0:
            return 0.5
        
        # Coefficient of variation, normalized to 0-1
        cv = std / mean
        consistency = max(0, 1 - cv)
        
        return float(consistency)
    
    def predict_next_bloom(self, 
                          cycles: List[Dict],
                          current_date: Optional[datetime] = None) -> Dict:
        """
        Predict the next bloom event based on historical cycles
        
        Args:
            cycles: List of detected bloom cycles
            current_date: Reference date for prediction (default: today)
            
        Returns:
            Dictionary with bloom prediction
        """
        if not cycles:
            return {
                'predicted': False,
                'message': 'Insufficient historical data for prediction'
            }
        
        if current_date is None:
            current_date = datetime.now()
        
        # Calculate seasonal pattern
        pattern = self.calculate_seasonal_pattern(cycles)
        
        if not pattern['has_pattern']:
            return {
                'predicted': False,
                'message': 'No clear bloom pattern detected'
            }
        
        # Predict next bloom based on average timing
        current_day_of_year = current_date.timetuple().tm_yday
        avg_peak_day = pattern['avg_peak_day']
        
        # Calculate days until next predicted bloom
        if current_day_of_year < avg_peak_day:
            days_until = avg_peak_day - current_day_of_year
            predicted_date = current_date + timedelta(days=days_until)
        else:
            # Next year's bloom
            days_until = (365 - current_day_of_year) + avg_peak_day
            predicted_date = current_date + timedelta(days=days_until)
        
        # Calculate confidence based on consistency
        confidence = pattern['consistency_score']
        
        # Predict bloom characteristics
        prediction = {
            'predicted': True,
            'predicted_date': predicted_date,
            'days_until_bloom': days_until,
            'confidence': confidence,
            'confidence_level': self._get_confidence_level(confidence),
            'predicted_peak_ndvi': pattern['avg_peak_ndvi'],
            'predicted_duration_days': pattern['avg_duration_days'],
            'typical_season': pattern.get('typical_season', 'Unknown'),
            'uncertainty_days': int(pattern['std_peak_day']),
            'based_on_cycles': pattern['num_cycles']
        }
        
        return prediction
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to categorical level"""
        if confidence >= 0.8:
            return 'High'
        elif confidence >= 0.6:
            return 'Moderate'
        elif confidence >= 0.4:
            return 'Low'
        else:
            return 'Very Low'
    
    def analyze_trend(self, 
                     dates: List[datetime],
                     ndvi_values: List[float]) -> Dict:
        """
        Analyze long-term trend in vegetation indices
        
        Args:
            dates: List of observation dates
            ndvi_values: Corresponding NDVI values
            
        Returns:
            Dictionary with trend analysis
        """
        if len(dates) < 5:
            return {
                'has_trend': False,
                'message': 'Insufficient data for trend analysis'
            }
        
        # Convert dates to numeric (days since first observation)
        dates_arr = np.array([d.timestamp() for d in dates])
        days = (dates_arr - dates_arr.min()) / (24 * 3600)
        
        ndvi_arr = np.array(ndvi_values).reshape(-1, 1)
        days_reshaped = days.reshape(-1, 1)
        
        # Fit linear regression
        self.model.fit(days_reshaped, ndvi_arr)
        slope = self.model.coef_[0][0]
        intercept = self.model.intercept_[0]
        score = self.model.score(days_reshaped, ndvi_arr)
        
        # Determine trend direction
        if abs(slope) < 0.0001:
            direction = 'Stable'
        elif slope > 0:
            direction = 'Increasing'
        else:
            direction = 'Decreasing'
        
        # Calculate rate of change (NDVI units per year)
        yearly_change = slope * 365
        
        trend = {
            'has_trend': True,
            'direction': direction,
            'slope': float(slope),
            'yearly_change': float(yearly_change),
            'r_squared': float(score),
            'confidence': float(score),
            'interpretation': self._interpret_trend(direction, yearly_change, score)
        }
        
        return trend
    
    def _interpret_trend(self, direction: str, yearly_change: float, r_squared: float) -> str:
        """Generate human-readable trend interpretation"""
        if r_squared < 0.3:
            return "No clear long-term trend detected. Vegetation shows high variability."
        
        if direction == 'Stable':
            return "Vegetation patterns are relatively stable over time."
        elif direction == 'Increasing':
            change_pct = abs(yearly_change) * 100
            if change_pct > 10:
                return f"Strong greening trend ({change_pct:.1f}% per year). Indicates improving vegetation health."
            else:
                return f"Moderate greening trend ({change_pct:.1f}% per year). Slight improvement in vegetation."
        else:  # Decreasing
            change_pct = abs(yearly_change) * 100
            if change_pct > 10:
                return f"Significant browning trend ({change_pct:.1f}% per year). May indicate stress or land use change."
            else:
                return f"Slight browning trend ({change_pct:.1f}% per year). Minor vegetation decline."
    
    def generate_forecast(self,
                         historical_data: pd.DataFrame,
                         forecast_days: int = 180) -> pd.DataFrame:
        """
        Generate vegetation index forecast for specified period
        
        Args:
            historical_data: DataFrame with 'date' and 'ndvi' columns
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with forecasted values
        """
        if len(historical_data) < 30:
            return pd.DataFrame()
        
        # Extract dates and values
        dates = pd.to_datetime(historical_data['date'])
        ndvi = historical_data['ndvi'].values
        
        # Detect cycles
        cycles = self.detect_bloom_cycles(dates.tolist(), ndvi.tolist())
        
        if not cycles:
            # Use simple trend-based forecast
            trend = self.analyze_trend(dates.tolist(), ndvi.tolist())
            
            last_date = dates.max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Linear projection
            days_ahead = np.arange(1, forecast_days + 1)
            if trend['has_trend']:
                forecast_ndvi = ndvi[-1] + (trend['slope'] * days_ahead)
            else:
                forecast_ndvi = np.full(forecast_days, ndvi[-1])
            
            forecast_ndvi = np.clip(forecast_ndvi, 0, 1)
            
        else:
            # Use cyclic pattern for forecast
            pattern = self.calculate_seasonal_pattern(cycles)
            
            last_date = dates.max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Create seasonal forecast
            forecast_ndvi = []
            for date in forecast_dates:
                day_of_year = date.timetuple().tm_yday
                
                # Distance from peak
                peak_day = pattern['avg_peak_day']
                distance_from_peak = min(
                    abs(day_of_year - peak_day),
                    365 - abs(day_of_year - peak_day)
                )
                
                # Gaussian-like bloom curve
                sigma = pattern['avg_duration_days'] / 3
                bloom_factor = np.exp(-(distance_from_peak ** 2) / (2 * sigma ** 2))
                
                # Baseline + bloom
                baseline = 0.3
                peak_increase = pattern['avg_peak_ndvi'] - baseline
                ndvi_forecast = baseline + (bloom_factor * peak_increase)
                
                forecast_ndvi.append(ndvi_forecast)
            
            forecast_ndvi = np.array(forecast_ndvi)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast_ndvi': forecast_ndvi,
            'lower_bound': forecast_ndvi * 0.85,
            'upper_bound': forecast_ndvi * 1.15
        })
        
        return forecast_df
    
    def calculate_bloom_probability(self,
                                   historical_data: pd.DataFrame,
                                   target_date: datetime) -> Dict:
        """
        Calculate probability of bloom event on target date
        
        Args:
            historical_data: DataFrame with historical bloom data
            target_date: Date to calculate bloom probability for
            
        Returns:
            Dictionary with probability and supporting information
        """
        if len(historical_data) < 10:
            return {
                'probability': 0.0,
                'confidence': 'Very Low',
                'message': 'Insufficient historical data'
            }
        
        # Extract dates and NDVI
        dates = pd.to_datetime(historical_data['date'])
        ndvi = historical_data['ndvi'].values
        
        # Detect historical cycles
        cycles = self.detect_bloom_cycles(dates.tolist(), ndvi.tolist())
        
        if not cycles:
            return {
                'probability': 0.0,
                'confidence': 'Low',
                'message': 'No clear bloom pattern detected'
            }
        
        # Calculate target day of year
        target_day = target_date.timetuple().tm_yday
        
        # Calculate distance to historical bloom dates
        bloom_days = [c['day_of_year'] for c in cycles]
        
        # Find closest historical bloom
        distances = [min(abs(target_day - bd), 365 - abs(target_day - bd)) 
                    for bd in bloom_days]
        min_distance = min(distances)
        
        # Probability decreases with distance
        # Within 7 days: high probability
        # Within 30 days: moderate probability
        # Beyond 30 days: low probability
        if min_distance <= 7:
            probability = 0.9
        elif min_distance <= 14:
            probability = 0.7
        elif min_distance <= 30:
            probability = 0.4
        elif min_distance <= 60:
            probability = 0.2
        else:
            probability = 0.1
        
        pattern = self.calculate_seasonal_pattern(cycles)
        probability *= pattern['consistency_score']
        
        return {
            'probability': float(probability),
            'confidence': self._get_confidence_level(pattern['consistency_score']),
            'days_to_nearest_bloom': int(min_distance),
            'historical_blooms': len(cycles),
            'pattern_consistency': pattern['consistency_score']
        }
