"""
Professional UI Components for BloomSphere
Provides reusable, consistently styled components for enterprise-grade UI
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

# NASA-inspired professional color palette
COLORS = {
    'primary': '#00A6D6',      # Teal
    'secondary': '#1E3A8A',    # Deep Indigo
    'accent': '#FF6B6B',       # Coral
    'success': '#10B981',      # Green
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'text_dark': '#1A202C',    # Dark slate
    'text_light': '#6B7280',   # Gray
    'bg_light': '#F5F7FA',     # Light blue-gray
    'bg_card': '#FFFFFF',      # White
}

# Spacing constants (8px grid system)
SPACING = {
    'xs': '0.5rem',   # 8px
    'sm': '1rem',     # 16px
    'md': '1.5rem',   # 24px
    'lg': '2rem',     # 32px
    'xl': '3rem',     # 48px
}


def page_header(title: str, subtitle: Optional[str] = None, icon: str = "🌸"):
    """
    Professional page header with title, optional subtitle, and icon
    
    Args:
        title: Page title
        subtitle: Optional subtitle text
        icon: Emoji icon for the page
    """
    st.markdown(f"""
        <div style="padding: {SPACING['md']} 0; border-bottom: 2px solid {COLORS['bg_light']}; margin-bottom: {SPACING['lg']};">
            <h1 style="margin: 0; color: {COLORS['text_dark']}; font-size: 2.5rem; font-weight: 700;">
                {icon} {title}
            </h1>
            {f'<p style="margin: {SPACING["xs"]} 0 0 0; color: {COLORS["text_light"]}; font-size: 1.1rem;">{subtitle}</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: Optional[str] = None, 
                delta_positive: bool = True, icon: str = "📊"):
    """
    Professional metric card with optional delta indicator
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional change indicator
        delta_positive: Whether delta is positive (green) or negative (red)
        icon: Emoji icon for the metric
    """
    delta_color = COLORS['success'] if delta_positive else COLORS['danger']
    delta_html = f'<div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.25rem;">{delta}</div>' if delta else ''
    
    st.markdown(f"""
        <div style="
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['bg_light']};
            border-radius: 0.5rem;
            padding: {SPACING['md']};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            height: 100%;
        ">
            <div style="color: {COLORS['text_light']}; font-size: 0.875rem; margin-bottom: {SPACING['xs']};">
                {icon} {label}
            </div>
            <div style="color: {COLORS['text_dark']}; font-size: 2rem; font-weight: 700;">
                {value}
            </div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: Optional[str] = None):
    """
    Section header for organizing content within pages
    
    Args:
        title: Section title
        subtitle: Optional description text
    """
    st.markdown(f"""
        <div style="margin: {SPACING['lg']} 0 {SPACING['md']} 0;">
            <h2 style="margin: 0; color: {COLORS['text_dark']}; font-size: 1.75rem; font-weight: 600;">
                {title}
            </h2>
            {f'<p style="margin: {SPACING["xs"]} 0 0 0; color: {COLORS["text_light"]};">{subtitle}</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def info_panel(content: str, panel_type: str = "info", icon: str = "ℹ️"):
    """
    Styled information panel
    
    Args:
        content: Panel content text
        panel_type: One of 'info', 'success', 'warning', 'danger'
        icon: Emoji icon
    """
    colors_map = {
        'info': {'bg': '#EFF6FF', 'border': COLORS['primary'], 'text': '#1E40AF'},
        'success': {'bg': '#F0FDF4', 'border': COLORS['success'], 'text': '#166534'},
        'warning': {'bg': '#FFFBEB', 'border': COLORS['warning'], 'text': '#92400E'},
        'danger': {'bg': '#FEF2F2', 'border': COLORS['danger'], 'text': '#991B1B'},
    }
    
    colors = colors_map.get(panel_type, colors_map['info'])
    
    st.markdown(f"""
        <div style="
            background: {colors['bg']};
            border-left: 4px solid {colors['border']};
            border-radius: 0.375rem;
            padding: {SPACING['md']};
            margin: {SPACING['md']} 0;
        ">
            <div style="color: {colors['text']}; font-size: 0.95rem;">
                {icon} {content}
            </div>
        </div>
    """, unsafe_allow_html=True)


def card_container(title: Optional[str] = None):
    """
    Context manager for creating a card-style container
    
    Args:
        title: Optional card title
    
    Usage:
        with card_container("My Card"):
            st.write("Card content")
    """
    container = st.container()
    with container:
        if title:
            st.markdown(f"""
                <div style="
                    background: {COLORS['bg_card']};
                    border: 1px solid {COLORS['bg_light']};
                    border-radius: 0.5rem;
                    padding: {SPACING['md']};
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    margin-bottom: {SPACING['md']};
                ">
                    <h3 style="margin: 0 0 {SPACING['md']} 0; color: {COLORS['text_dark']}; font-size: 1.25rem; font-weight: 600;">
                        {title}
                    </h3>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="
                    background: {COLORS['bg_card']};
                    border: 1px solid {COLORS['bg_light']};
                    border-radius: 0.5rem;
                    padding: {SPACING['md']};
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    margin-bottom: {SPACING['md']};
                ">
            """, unsafe_allow_html=True)
    
    return container


def feature_card(icon: str, title: str, description: str):
    """
    Feature/value proposition card
    
    Args:
        icon: Emoji icon
        title: Feature title
        description: Feature description
    """
    st.markdown(f"""
        <div style="
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['bg_light']};
            border-radius: 0.5rem;
            padding: {SPACING['lg']};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            transition: transform 0.2s;
        ">
            <div style="font-size: 3rem; margin-bottom: {SPACING['md']};">
                {icon}
            </div>
            <h3 style="margin: 0 0 {SPACING['sm']} 0; color: {COLORS['text_dark']}; font-size: 1.25rem; font-weight: 600;">
                {title}
            </h3>
            <p style="margin: 0; color: {COLORS['text_light']}; line-height: 1.6;">
                {description}
            </p>
        </div>
    """, unsafe_allow_html=True)


def create_professional_chart_layout() -> Dict[str, Any]:
    """
    Returns a professional Plotly chart layout configuration
    """
    return {
        'font': {'family': 'sans-serif', 'size': 12, 'color': COLORS['text_dark']},
        'plot_bgcolor': COLORS['bg_card'],
        'paper_bgcolor': COLORS['bg_card'],
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
        'xaxis': {
            'gridcolor': COLORS['bg_light'],
            'linecolor': COLORS['bg_light'],
            'showgrid': True,
        },
        'yaxis': {
            'gridcolor': COLORS['bg_light'],
            'linecolor': COLORS['bg_light'],
            'showgrid': True,
        },
        'colorway': [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                     COLORS['success'], COLORS['warning']],
    }


def loading_message(message: str = "Processing..."):
    """
    Display a professional loading indicator
    
    Args:
        message: Loading message text
    """
    st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            padding: {SPACING['xl']};
            color: {COLORS['text_light']};
        ">
            <div style="margin-right: {SPACING['md']};">
                ⏳
            </div>
            <div style="font-size: 1.1rem;">
                {message}
            </div>
        </div>
    """, unsafe_allow_html=True)


def empty_state(icon: str, title: str, message: str, action_text: Optional[str] = None):
    """
    Professional empty state display
    
    Args:
        icon: Emoji icon
        title: Empty state title
        message: Description message
        action_text: Optional call-to-action text
    """
    action_html = f'<p style="margin-top: {SPACING["md"]}; color: {COLORS["primary"]}; font-weight: 600;">{action_text}</p>' if action_text else ''
    
    st.markdown(f"""
        <div style="
            text-align: center;
            padding: {SPACING['xl']} {SPACING['md']};
            color: {COLORS['text_light']};
        ">
            <div style="font-size: 4rem; margin-bottom: {SPACING['md']}; opacity: 0.5;">
                {icon}
            </div>
            <h3 style="margin: 0 0 {SPACING['sm']} 0; color: {COLORS['text_dark']};">
                {title}
            </h3>
            <p style="margin: 0; max-width: 500px; margin-left: auto; margin-right: auto;">
                {message}
            </p>
            {action_html}
        </div>
    """, unsafe_allow_html=True)


def divider(spacing: str = 'md'):
    """
    Visual divider with consistent spacing
    
    Args:
        spacing: Size from SPACING dict ('xs', 'sm', 'md', 'lg', 'xl')
    """
    space = SPACING.get(spacing, SPACING['md'])
    st.markdown(f"""
        <div style="
            border-top: 1px solid {COLORS['bg_light']};
            margin: {space} 0;
        "></div>
    """, unsafe_allow_html=True)


def hero_section(title: str, subtitle: str, icon: str = "🌸"):
    """
    Hero banner for landing page
    
    Args:
        title: Main hero title
        subtitle: Hero subtitle/description
        icon: Hero icon
    """
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%);
            color: white;
            padding: {SPACING['xl']} {SPACING['lg']};
            border-radius: 0.75rem;
            margin-bottom: {SPACING['lg']};
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 4rem; margin-bottom: {SPACING['md']};">
                {icon}
            </div>
            <h1 style="margin: 0 0 {SPACING['md']} 0; font-size: 3rem; font-weight: 700;">
                {title}
            </h1>
            <p style="margin: 0; font-size: 1.25rem; opacity: 0.95; max-width: 800px; margin-left: auto; margin-right: auto;">
                {subtitle}
            </p>
        </div>
    """, unsafe_allow_html=True)
