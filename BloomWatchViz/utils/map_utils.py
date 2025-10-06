import numpy as np
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import json
from datetime import datetime, timedelta

class MapUtils:
    """
    Utility class for creating and managing interactive maps for bloom visualization.
    Provides functions for map creation, styling, data overlay, and interaction handling.
    """
    
    def __init__(self):
        """Initialize the MapUtils class with default configurations."""
        
        # Default map configurations
        self.default_config = {
            'center_lat': 20.0,
            'center_lon': 0.0,
            'zoom_start': 3,
            'min_zoom': 2,
            'max_zoom': 18,
            'prefer_canvas': True
        }
        
        # Color schemes for different visualizations
        self.color_schemes = {
            'bloom_intensity': {
                'colors': ['#FFFACD', '#F0E68C', '#9ACD32', '#32CD32', '#006400'],
                'values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                'labels': ['Minimal', 'Low', 'Moderate', 'High', 'Peak']
            },
            'vegetation_health': {
                'colors': ['#8B0000', '#FF4500', '#FFA500', '#90EE90', '#228B22'],
                'values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                'labels': ['Poor', 'Fair', 'Moderate', 'Good', 'Excellent']
            },
            'temporal_change': {
                'colors': ['#0000FF', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF0000'],
                'values': [-1.0, -0.5, 0.0, 0.5, 1.0],
                'labels': ['Strong Decrease', 'Decrease', 'No Change', 'Increase', 'Strong Increase']
            }
        }
        
        # Tile layer options
        self.tile_layers = {
            'openstreetmap': 'OpenStreetMap',
            'satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'terrain': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            'toner': 'https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
            'watercolor': 'https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg'
        }
    
    def create_base_map(self, center_lat: float = None, center_lon: float = None,
                       zoom_start: int = None, tile_layer: str = 'openstreetmap',
                       **kwargs) -> folium.Map:
        """
        Create a base folium map with specified parameters.
        
        Args:
            center_lat (float): Center latitude for the map
            center_lon (float): Center longitude for the map
            zoom_start (int): Initial zoom level
            tile_layer (str): Tile layer to use ('openstreetmap', 'satellite', 'terrain', etc.)
            **kwargs: Additional parameters for folium.Map()
            
        Returns:
            folium.Map: Configured folium map object
        """
        
        # Use default values if not provided
        center_lat = center_lat or self.default_config['center_lat']
        center_lon = center_lon or self.default_config['center_lon']
        zoom_start = zoom_start or self.default_config['zoom_start']
        
        # Get tile layer URL
        tiles = self.tile_layers.get(tile_layer, tile_layer)
        
        # Create map
        map_obj = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tiles if tiles in ['OpenStreetMap', 'Stamen Terrain', 'Stamen Toner'] else None,
            min_zoom=self.default_config['min_zoom'],
            max_zoom=self.default_config['max_zoom'],
            prefer_canvas=self.default_config['prefer_canvas'],
            **kwargs
        )
        
        # Add custom tile layer if needed
        if tiles not in ['OpenStreetMap', 'Stamen Terrain', 'Stamen Toner']:
            folium.TileLayer(
                tiles=tiles,
                attr='Custom Tile Layer',
                name=tile_layer,
                overlay=False,
                control=True
            ).add_to(map_obj)
        
        return map_obj
    
    def add_bloom_markers(self, map_obj: folium.Map, bloom_data: pd.DataFrame,
                         intensity_column: str = 'intensity',
                         color_scheme: str = 'bloom_intensity') -> folium.Map:
        """
        Add bloom detection markers to the map.
        
        Args:
            map_obj (folium.Map): Folium map object
            bloom_data (pd.DataFrame): DataFrame with bloom data (must have 'lat', 'lon' columns)
            intensity_column (str): Column name for bloom intensity values
            color_scheme (str): Color scheme to use for markers
            
        Returns:
            folium.Map: Map with added bloom markers
        """
        
        if bloom_data.empty:
            return map_obj
        
        # Get color scheme
        scheme = self.color_schemes.get(color_scheme, self.color_schemes['bloom_intensity'])
        
        # Add markers for each bloom location
        for idx, row in bloom_data.iterrows():
            try:
                # Get color based on intensity
                color = self.get_color_from_value(row[intensity_column], scheme)
                
                # Create popup content
                popup_content = self.create_bloom_popup(row, intensity_column)
                
                # Add circle marker
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Bloom Intensity: {row[intensity_column]:.3f}",
                    color='white',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(map_obj)
                
            except Exception as e:
                print(f"Error adding marker for row {idx}: {e}")
                continue
        
        return map_obj
    
    def add_heatmap_layer(self, map_obj: folium.Map, data_points: List[List[float]],
                         name: str = 'Bloom Heatmap', radius: int = 15,
                         blur: int = 10, min_opacity: float = 0.4) -> folium.Map:
        """
        Add a heatmap layer to the map.
        
        Args:
            map_obj (folium.Map): Folium map object
            data_points (List[List[float]]): List of [lat, lon, intensity] points
            name (str): Name for the heatmap layer
            radius (int): Radius of each heat point
            blur (int): Blur factor for heat points
            min_opacity (float): Minimum opacity for the heatmap
            
        Returns:
            folium.Map: Map with added heatmap layer
        """
        
        if not data_points:
            return map_obj
        
        # Create heatmap layer
        heatmap = plugins.HeatMap(
            data_points,
            name=name,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            gradient={
                0.0: '#000080',  # Dark blue
                0.2: '#0000FF',  # Blue
                0.4: '#00FF00',  # Green
                0.6: '#FFFF00',  # Yellow
                0.8: '#FF8000',  # Orange
                1.0: '#FF0000'   # Red
            }
        )
        
        heatmap.add_to(map_obj)
        
        return map_obj
    
    def create_choropleth_map(self, map_obj: folium.Map, geojson_data: Dict,
                            data_df: pd.DataFrame, columns: List[str],
                            key_on: str = 'feature.properties.NAME',
                            fill_color: str = 'YlOrRd',
                            legend_name: str = 'Bloom Intensity') -> folium.Map:
        """
        Create a choropleth map for regional bloom data.
        
        Args:
            map_obj (folium.Map): Folium map object
            geojson_data (Dict): GeoJSON data for regions
            data_df (pd.DataFrame): DataFrame with regional data
            columns (List[str]): [key_column, data_column] for choropleth
            key_on (str): Key path in GeoJSON properties
            fill_color (str): Color scheme for choropleth
            legend_name (str): Name for the legend
            
        Returns:
            folium.Map: Map with choropleth layer
        """
        
        try:
            folium.Choropleth(
                geo_data=geojson_data,
                name='choropleth',
                data=data_df,
                columns=columns,
                key_on=key_on,
                fill_color=fill_color,
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=legend_name
            ).add_to(map_obj)
            
        except Exception as e:
            print(f"Error creating choropleth map: {e}")
        
        return map_obj
    
    def add_legend(self, map_obj: folium.Map, color_scheme: str = 'bloom_intensity',
                  title: str = 'Bloom Intensity') -> folium.Map:
        """
        Add a custom legend to the map.
        
        Args:
            map_obj (folium.Map): Folium map object
            color_scheme (str): Color scheme to use for legend
            title (str): Title for the legend
            
        Returns:
            folium.Map: Map with added legend
        """
        
        scheme = self.color_schemes.get(color_scheme, self.color_schemes['bloom_intensity'])
        
        # Create legend HTML
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;">
        <h4 style="margin-top:0; color: #333;">{title}</h4>
        '''
        
        # Add color entries
        for i, (color, label) in enumerate(zip(scheme['colors'], scheme['labels'])):
            legend_html += f'''
            <p style="margin: 5px 0;">
                <i style="background:{color}; width:18px; height:18px; 
                   float:left; margin-right:8px; border-radius: 3px; 
                   border: 1px solid #ccc;"></i> 
                {label}
            </p>
            '''
        
        legend_html += '</div>'
        
        # Add legend to map
        map_obj.get_root().html.add_child(folium.Element(legend_html))
        
        return map_obj
    
    def get_color_from_value(self, value: float, color_scheme: Dict) -> str:
        """
        Get color for a value based on the color scheme.
        
        Args:
            value (float): Value to get color for
            color_scheme (Dict): Color scheme configuration
            
        Returns:
            str: Hex color code
        """
        
        if np.isnan(value):
            return '#CCCCCC'  # Gray for NaN values
        
        colors = color_scheme['colors']
        values = color_scheme['values']
        
        # Find the appropriate color
        for i in range(len(values) - 1):
            if values[i] <= value <= values[i + 1]:
                # Linear interpolation between colors if needed
                return colors[i]
        
        # Return last color if value exceeds range
        if value > values[-1]:
            return colors[-1]
        else:
            return colors[0]
    
    def create_bloom_popup(self, data_row: pd.Series, intensity_column: str) -> str:
        """
        Create HTML popup content for bloom markers.
        
        Args:
            data_row (pd.Series): Row of bloom data
            intensity_column (str): Column name for intensity values
            
        Returns:
            str: HTML content for popup
        """
        
        # Extract available information
        lat = data_row.get('lat', 'N/A')
        lon = data_row.get('lon', 'N/A')
        intensity = data_row.get(intensity_column, 'N/A')
        date = data_row.get('date', 'N/A')
        confidence = data_row.get('confidence', 'N/A')
        land_cover = data_row.get('land_cover', 'N/A')
        bloom_stage = data_row.get('bloom_stage', 'N/A')
        
        popup_html = f'''
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #2E8B57; border-bottom: 1px solid #ccc; padding-bottom: 5px;">
                🌸 Bloom Detection
            </h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td style="font-weight: bold; padding: 2px;">Location:</td><td style="padding: 2px;">{lat:.4f}°N, {lon:.4f}°E</td></tr>
                <tr><td style="font-weight: bold; padding: 2px;">Intensity:</td><td style="padding: 2px;">{intensity:.3f if isinstance(intensity, (int, float)) else intensity}</td></tr>
        '''
        
        if date != 'N/A':
            popup_html += f'<tr><td style="font-weight: bold; padding: 2px;">Date:</td><td style="padding: 2px;">{date}</td></tr>'
        
        if confidence != 'N/A':
            popup_html += f'<tr><td style="font-weight: bold; padding: 2px;">Confidence:</td><td style="padding: 2px;">{confidence}%</td></tr>'
        
        if land_cover != 'N/A':
            popup_html += f'<tr><td style="font-weight: bold; padding: 2px;">Land Cover:</td><td style="padding: 2px;">{land_cover}</td></tr>'
        
        if bloom_stage != 'N/A':
            popup_html += f'<tr><td style="font-weight: bold; padding: 2px;">Bloom Stage:</td><td style="padding: 2px;">{bloom_stage}</td></tr>'
        
        popup_html += '''
            </table>
        </div>
        '''
        
        return popup_html
    
    def create_plotly_map(self, data: pd.DataFrame, lat_col: str = 'lat', 
                         lon_col: str = 'lon', color_col: str = 'intensity',
                         size_col: str = None, title: str = 'Bloom Activity Map') -> go.Figure:
        """
        Create an interactive Plotly map visualization.
        
        Args:
            data (pd.DataFrame): Data to plot
            lat_col (str): Column name for latitude
            lon_col (str): Column name for longitude  
            color_col (str): Column name for color values
            size_col (str): Column name for marker sizes (optional)
            title (str): Title for the map
            
        Returns:
            go.Figure: Plotly figure object
        """
        
        # Create hover text
        hover_text = []
        for _, row in data.iterrows():
            text = f"Location: {row[lat_col]:.3f}°, {row[lon_col]:.3f}°<br>"
            text += f"Intensity: {row[color_col]:.3f}<br>"
            
            # Add additional information if available
            if 'date' in data.columns:
                text += f"Date: {row['date']}<br>"
            if 'confidence' in data.columns:
                text += f"Confidence: {row['confidence']}%<br>"
            if 'land_cover' in data.columns:
                text += f"Land Cover: {row['land_cover']}"
                
            hover_text.append(text)
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scattergeo(
            lon=data[lon_col],
            lat=data[lat_col],
            mode='markers',
            marker=dict(
                size=data[size_col] if size_col and size_col in data.columns else 8,
                color=data[color_col],
                colorscale='Viridis',
                colorbar=dict(
                    title=color_col.title(),
                    titleside="right"
                ),
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Bloom Activity'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 245, 255)',
                showlakes=True,
                lakecolor='rgb(230, 245, 255)',
                showrivers=True,
                rivercolor='rgb(230, 245, 255)'
            ),
            margin=dict(t=50, b=0, l=0, r=0),
            height=600
        )
        
        return fig
    
    def create_temporal_map_animation(self, data: pd.DataFrame, date_col: str = 'date',
                                    lat_col: str = 'lat', lon_col: str = 'lon',
                                    color_col: str = 'intensity') -> go.Figure:
        """
        Create an animated temporal map showing bloom progression over time.
        
        Args:
            data (pd.DataFrame): Time series data with bloom information
            date_col (str): Column name for dates
            lat_col (str): Column name for latitude
            lon_col (str): Column name for longitude
            color_col (str): Column name for color values
            
        Returns:
            go.Figure: Animated Plotly figure
        """
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data = data.sort_values(date_col)
        
        # Create figure
        fig = px.scatter_geo(
            data,
            lat=lat_col,
            lon=lon_col,
            color=color_col,
            animation_frame=data[date_col].dt.strftime('%Y-%m-%d'),
            title='Bloom Activity Animation',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
            ),
            height=600
        )
        
        return fig
    
    def calculate_map_bounds(self, data: pd.DataFrame, lat_col: str = 'lat', 
                           lon_col: str = 'lon', padding: float = 0.1) -> Dict[str, float]:
        """
        Calculate appropriate map bounds for data points.
        
        Args:
            data (pd.DataFrame): Data with location information
            lat_col (str): Column name for latitude
            lon_col (str): Column name for longitude
            padding (float): Padding factor for bounds
            
        Returns:
            Dict[str, float]: Dictionary with bounds information
        """
        
        if data.empty:
            return {
                'min_lat': -60, 'max_lat': 70,
                'min_lon': -180, 'max_lon': 180,
                'center_lat': 0, 'center_lon': 0
            }
        
        min_lat, max_lat = data[lat_col].min(), data[lat_col].max()
        min_lon, max_lon = data[lon_col].min(), data[lon_col].max()
        
        # Add padding
        lat_padding = (max_lat - min_lat) * padding
        lon_padding = (max_lon - min_lon) * padding
        
        bounds = {
            'min_lat': max(min_lat - lat_padding, -90),
            'max_lat': min(max_lat + lat_padding, 90),
            'min_lon': max(min_lon - lon_padding, -180),
            'max_lon': min(max_lon + lon_padding, 180),
            'center_lat': (min_lat + max_lat) / 2,
            'center_lon': (min_lon + max_lon) / 2
        }
        
        return bounds
    
    def add_map_controls(self, map_obj: folium.Map) -> folium.Map:
        """
        Add various controls to the map (layer control, fullscreen, etc.).
        
        Args:
            map_obj (folium.Map): Folium map object
            
        Returns:
            folium.Map: Map with added controls
        """
        
        # Add layer control
        folium.LayerControl().add_to(map_obj)
        
        # Add fullscreen button
        plugins.Fullscreen(
            position='topright',
            title='Fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(map_obj)
        
        # Add measure control
        plugins.MeasureControl(
            primary_length_unit='kilometers',
            secondary_length_unit='miles',
            primary_area_unit='sqkilometers',
            secondary_area_unit='sqmiles'
        ).add_to(map_obj)
        
        # Add draw control
        draw = plugins.Draw(
            export=True,
            filename='bloom_analysis_area.geojson',
            position='topleft'
        )
        draw.add_to(map_obj)
        
        return map_obj
    
    def export_map(self, map_obj: folium.Map, filename: str, format_type: str = 'html') -> bool:
        """
        Export map to file.
        
        Args:
            map_obj (folium.Map): Folium map object to export
            filename (str): Output filename
            format_type (str): Export format ('html', 'png')
            
        Returns:
            bool: True if export successful
        """
        
        try:
            if format_type.lower() == 'html':
                map_obj.save(filename)
                return True
            elif format_type.lower() == 'png':
                # Note: PNG export requires additional dependencies like selenium
                print("PNG export requires additional setup with selenium and webdriver")
                return False
            else:
                print(f"Unsupported export format: {format_type}")
                return False
                
        except Exception as e:
            print(f"Error exporting map: {e}")
            return False
