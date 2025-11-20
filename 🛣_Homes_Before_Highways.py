# Standard libraries
import os
import time
import json
import logging
from datetime import datetime
import io
from io import BytesIO
import base64

# Data handling
import pandas as pd
import numpy as np
import geopandas as gpd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from wordcloud import WordCloud
import pydeck as pdk

# Mapping
import folium
from folium.plugins import MeasureControl, MarkerCluster, HeatMap
from branca.colormap import linear
from streamlit_folium import folium_static, st_folium

# Image & animation
from PIL import Image, UnidentifiedImageError
import requests

# Streamlit core and components
import streamlit as st

# Set page configuration - MAKE WIDE BY DEFAULT
st.set_page_config(
    page_title="Homes Before Highways", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🏠"
)

# Enhanced CSS for scrolling design
st.markdown("""
<style>
    /* Global Styles - Full screen */
    .stApp {
        background: #F9FAF9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1C1C1C;
    }
    
    /* Main content area - maximize space */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #007A33 0%, #005924 100%);
        color: white;
        padding: 2rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    /* Filter section */
    .filter-container {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .filter-title {
        color: #007A33;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #E0E4E2;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #F3F6F4 0%, #E6F4EC 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .metric-card:hover {
        border: 2px solid #007A33;
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #007A33;
        margin-bottom: 0.5rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-subtitle { font-size: 1rem; }
        .metric-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data loading with performance optimization
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("highway_projects_with_districts.csv")
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Num_Home_Demolished', 'Num_Business_Demolished', 'Total_Relocations']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Extract coordinates and calculate project length more efficiently
        if 'Project Start Point' in df.columns and 'Project End Point' in df.columns:
            def extract_coords(coord_str):
                if pd.isna(coord_str):
                    return pd.NA, pd.NA
                try:
                    parts = coord_str.replace('"', '').strip().split(',')
                    return float(parts[0].strip()), float(parts[1].strip())
                except (ValueError, IndexError):
                    return pd.NA, pd.NA
            
            # Vectorized coordinate extraction
            start_coords = df['Project Start Point'].apply(extract_coords)
            end_coords = df['Project End Point'].apply(extract_coords)
            
            df['start_lat'] = [coord[0] for coord in start_coords]
            df['start_lng'] = [coord[1] for coord in start_coords]
            df['end_lat'] = [coord[0] for coord in end_coords]
            df['end_lng'] = [coord[1] for coord in end_coords]
            
            # Calculate project centers for mapping
            df['latitude'] = df[['start_lat', 'end_lat']].mean(axis=1)
            df['longitude'] = df[['start_lng', 'end_lng']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load map data with better caching
@st.cache_data(ttl=7200)
def load_map_data():
    try:
        # Define file paths
        LOCAL_GEOJSON_FILES = {
            "assembly": "California Assembly Districts.geojson",
            "senate": "California Senate Districts.geojson"
        }
        CALENVIRO_SHAPEFILE = "CalEnviroScreen_4.0.shp"
        HIGHWAYS_SHAPEFILE = "tl_2019_06_prisecroads.shp"
        
        # Initialize GeoDataFrames
        gdfs = {}
        
        # Load each file if it exists
        file_configs = {
            'assembly_gdf': {'file': LOCAL_GEOJSON_FILES["assembly"], 'simplify': 0.005},
            'senate_gdf': {'file': LOCAL_GEOJSON_FILES["senate"], 'simplify': 0.005},
            'calenviro_gdf': {'file': CALENVIRO_SHAPEFILE, 'simplify': 0.005, 
                             'columns': ['geometry', 'CES4.0Score', 'CIscoreP', 'TRACT']},
            'highways_gdf': {'file': HIGHWAYS_SHAPEFILE, 'simplify': 0.01}
        }
        
        for gdf_name, config in file_configs.items():
            try:
                if os.path.exists(config['file']):
                    if 'columns' in config:
                        gdf = gpd.read_file(config['file'], columns=config['columns'])
                    else:
                        gdf = gpd.read_file(config['file'])
                    
                    gdf['geometry'] = gdf.geometry.simplify(tolerance=config['simplify'])
                    
                    if gdf_name == 'calenviro_gdf':
                        if 'CES4.0Score' in gdf.columns:
                            gdf['CES_SCORE'] = gdf['CES4.0Score']
                        elif 'CIscoreP' in gdf.columns:
                            gdf['CES_SCORE'] = gdf['CIscoreP']
                        gdf = gdf[gdf.geometry.is_valid]
                    
                    gdfs[gdf_name] = gdf
                    logger.info(f"Successfully loaded {gdf_name}")
                else:
                    gdfs[gdf_name] = gpd.GeoDataFrame()
                    logger.warning(f"File not found: {config['file']}")
            except Exception as e:
                gdfs[gdf_name] = gpd.GeoDataFrame()
                logger.warning(f"Error loading {gdf_name}: {e}")
        
        return gdfs
    except Exception as e:
        logger.error(f"Error in load_map_data: {e}")
        return {key: gpd.GeoDataFrame() for key in ['assembly_gdf', 'senate_gdf', 'calenviro_gdf', 'highways_gdf']}

# Generate distinct colors for districts
def get_district_color(district_num, total_districts, base_hue='green'):
    """Generate distinct colors for districts"""
    if base_hue == 'green':
        hues = np.linspace(120, 150, total_districts)
    else:
        hues = np.linspace(200, 240, total_districts)
    
    idx = district_num % len(hues)
    h = hues[idx]
    s = 60 + (district_num % 3) * 15
    l = 45 + (district_num % 4) * 10
    
    c = (1 - abs(2 * l / 100 - 1)) * s / 100
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l / 100 - c / 2
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    return f'#{r:02x}{g:02x}{b:02x}'

# Load data
df = load_data()
map_data = load_map_data()

# Initialize session state to prevent reruns on map interactions
if 'map_rendered' not in st.session_state:
    st.session_state.map_rendered = False

# SHOW MAP TITLE FIRST
st.markdown("### 🗺️ Highway Projects Map")

# We'll create a placeholder for the map and update it after getting filter values
map_placeholder = st.empty()

# SHOW FILTERS BELOW
st.markdown('<div class="filter-title">🔍 Explore & Filter Highway Projects</div>', unsafe_allow_html=True)

# All filters in two rows
filter_row1 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

with filter_row1[0]:
    if not df.empty:
        assembly_cols = [col for col in df.columns if col.startswith('Assemblymember')]
        all_assembly = []
        for col in assembly_cols:
            all_assembly.extend(df[col].dropna().unique().tolist())
        all_assembly = sorted(list(set(all_assembly)))
        assembly_members = ["All Assembly Members"] + all_assembly
        selected_assembly = st.selectbox("🏛️ Assembly Member", assembly_members, key="unified_assembly")
    else:
        selected_assembly = "All Assembly Members"

with filter_row1[1]:
    if not df.empty:
        senator_cols = [col for col in df.columns if col.startswith('Senator')]
        all_senators = []
        for col in senator_cols:
            all_senators.extend(df[col].dropna().unique().tolist())
        all_senators = sorted(list(set(all_senators)))
        senators = ["All Senators"] + all_senators
        selected_senate = st.selectbox("🏛️ Senator", senators, key="unified_senate")
    else:
        selected_senate = "All Senators"

with filter_row1[2]:
    if not df.empty:
        counties = ["All Counties"] + sorted(df["County"].dropna().unique().tolist())
        selected_county = st.selectbox("📍 County", counties, key="unified_county")
    else:
        selected_county = "All Counties"

with filter_row1[3]:
    if not df.empty:
        routes = ["All Routes"] + sorted(df["Route"].dropna().astype(str).unique().tolist(), key=lambda x: (not x.isdigit(), x))
        selected_route = st.selectbox("🛣️ Route", routes, key="unified_route")
    else:
        selected_route = "All Routes"

with filter_row1[4]:
    if not df.empty:
        years = ["All Years"] + sorted(df["CCA_FY"].dropna().astype(str).unique().tolist())
        selected_year = st.selectbox("📅 Year", years, key="unified_year")
    else:
        selected_year = "All Years"

with filter_row1[5]:
    impact_filter = st.selectbox("🎯 Impact Level",
                                ["All Projects", "High Impact (100+)",
                                 "Medium Impact (20-100)", "Low Impact (1-20)", "No Impact"],
                                key="unified_impact")

filter_row2 = st.columns([2, 1.5, 5])

with filter_row2[0]:
    sort_by = st.selectbox("📊 Sort By",
                          ["Total Relocations", "Homes Demolished", "Businesses Demolished", 
                           "Year", "County", "Route"],
                          key="unified_sort")

st.markdown("#### 🎨 Map Layers")
layer_cols = st.columns(4)

with layer_cols[0]:
    show_assembly = st.checkbox("🏛️ Assembly Districts", value=True)
with layer_cols[1]:
    show_senate = st.checkbox("🏛️ Senate Districts", value=True)
with layer_cols[2]:
    show_highways = st.checkbox("Highways", value=False)

st.markdown('</div>', unsafe_allow_html=True)

# Apply all filters
if not df.empty:
    filtered_df = df.copy()
    
    if selected_assembly != "All Assembly Members":
        assembly_cols = [col for col in filtered_df.columns if col.startswith('Assemblymember')]
        mask = filtered_df[assembly_cols].apply(lambda row: selected_assembly in row.values, axis=1)
        filtered_df = filtered_df[mask]
    
    if selected_senate != "All Senators":
        senator_cols = [col for col in filtered_df.columns if col.startswith('Senator')]
        mask = filtered_df[senator_cols].apply(lambda row: selected_senate in row.values, axis=1)
        filtered_df = filtered_df[mask]
    
    if selected_county != "All Counties":
        filtered_df = filtered_df[filtered_df["County"] == selected_county]
    
    if selected_route != "All Routes":
        try:
            sel_num = int(selected_route)
            filtered_df = filtered_df[filtered_df["Route"] == sel_num]
        except:
            filtered_df = filtered_df[filtered_df["Route"].astype(str) == selected_route]
    
    if selected_year != "All Years":
        try:
            sel_year = int(selected_year)
            filtered_df = filtered_df[filtered_df["CCA_FY"] == sel_year]
        except:
            filtered_df = filtered_df[filtered_df["CCA_FY"].astype(str) == selected_year]
    
    if impact_filter != "All Projects":
        if impact_filter == "High Impact (100+)":
            filtered_df = filtered_df[filtered_df["Total_Relocations"].fillna(0) >= 100]
        elif impact_filter == "Medium Impact (20-100)":
            filtered_df = filtered_df[(filtered_df["Total_Relocations"].fillna(0) >= 20) &
                                     (filtered_df["Total_Relocations"].fillna(0) < 100)]
        elif impact_filter == "Low Impact (1-20)":
            filtered_df = filtered_df[(filtered_df["Total_Relocations"].fillna(0) >= 1) &
                                     (filtered_df["Total_Relocations"].fillna(0) < 20)]
        elif impact_filter == "No Impact":
            filtered_df = filtered_df[filtered_df["Total_Relocations"].fillna(0) == 0]
    
    if sort_by == "Total Relocations":
        filtered_df = filtered_df.sort_values("Total_Relocations", ascending=False)
    elif sort_by == "Homes Demolished":
        filtered_df = filtered_df.sort_values("Num_Home_Demolished", ascending=False)
    elif sort_by == "Businesses Demolished":
        filtered_df = filtered_df.sort_values("Num_Business_Demolished", ascending=False)
    elif sort_by == "Year":
        filtered_df = filtered_df.sort_values("CCA_FY", ascending=False)
    elif sort_by == "County":
        filtered_df = filtered_df.sort_values("County")
    elif sort_by == "Route":
        filtered_df = filtered_df.sort_values("Route")
    
    # CREATE MAP WITH FILTERED DATA
    if not filtered_df[filtered_df['latitude'].notnull()].empty:
        center_lat = float(filtered_df.loc[filtered_df['latitude'].notnull(), 'latitude'].mean())
        center_lng = float(filtered_df.loc[filtered_df['longitude'].notnull(), 'longitude'].mean())
        zoom_start = 6
    else:
        center_lat, center_lng, zoom_start = 37.2, -119.5, 6
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles='CartoDB positron',
        prefer_canvas=True
    )
    
    def _gdf_to_geojson(gdf):
        try:
            if hasattr(gdf, "to_json"):
                return json.loads(gdf.to_json())
            elif isinstance(gdf, dict):
                return gdf
        except Exception:
            return None
    
    if show_assembly and 'assembly_gdf' in map_data and not map_data['assembly_gdf'].empty:
        assembly_gdf = map_data['assembly_gdf']
        geojson_obj = _gdf_to_geojson(assembly_gdf)
        if geojson_obj and geojson_obj.get("features"):
            district_field = None
            for field in ['DISTRICT', 'District', 'district', 'AD', 'NAME', 'NAMELSAD', 'SLDLST', 'GEOID']:
                if field in assembly_gdf.columns:
                    district_field = field
                    break
            
            if not district_field and len(assembly_gdf.columns) > 1:
                cols = [c for c in assembly_gdf.columns if c != 'geometry']
                if cols:
                    district_field = cols[0]
            
            total_districts = len(assembly_gdf)
            assembly_colors = {}
            for idx, row in assembly_gdf.iterrows():
                if district_field:
                    try:
                        dist_val = row[district_field]
                        dist_num = int(float(str(dist_val).replace('Assembly District ', '').replace('AD', '').strip()))
                        assembly_colors[dist_val] = get_district_color(dist_num, 80, 'green')
                    except:
                        assembly_colors[row[district_field]] = '#90EE90'
            
            def assembly_style_function(feature):
                props = feature.get('properties', {}) or {}
                if district_field:
                    dist_val = props.get(district_field, 'Unknown')
                    color = assembly_colors.get(dist_val, '#90EE90')
                else:
                    color = '#90EE90'
                
                return {
                    'fillColor': color,
                    'color': '#006622',
                    'weight': 2.5,
                    'opacity': 0.9,
                    'fillOpacity': 0.5
                }
            
            tooltip_fields = [district_field] if district_field else []
            tooltip_aliases = ['Assembly District:'] if district_field else []
            
            folium.GeoJson(
                data=geojson_obj,
                name="Assembly Districts",
                style_function=assembly_style_function,
                highlight_function=lambda x: {'weight': 4, 'fillOpacity': 0.7},
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="background-color: white; color: #111827; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);"
                ) if tooltip_fields else None
            ).add_to(m)
    
    if show_senate and 'senate_gdf' in map_data and not map_data['senate_gdf'].empty:
        senate_gdf = map_data['senate_gdf']
        geojson_obj = _gdf_to_geojson(senate_gdf)
        if geojson_obj and geojson_obj.get("features"):
            district_field = None
            for field in ['DISTRICT', 'District', 'district', 'SD', 'NAME', 'NAMELSAD', 'SLDUST', 'GEOID']:
                if field in senate_gdf.columns:
                    district_field = field
                    break
            
            if not district_field and len(senate_gdf.columns) > 1:
                cols = [c for c in senate_gdf.columns if c != 'geometry']
                if cols:
                    district_field = cols[0]
            
            total_districts = len(senate_gdf)
            senate_colors = {}
            for idx, row in senate_gdf.iterrows():
                if district_field:
                    try:
                        dist_val = row[district_field]
                        dist_num = int(float(str(dist_val).replace('Senate District ', '').replace('SD', '').strip()))
                        senate_colors[dist_val] = get_district_color(dist_num, 40, 'blue')
                    except:
                        senate_colors[row[district_field]] = '#87CEEB'
            
            def senate_style_function(feature):
                props = feature.get('properties', {}) or {}
                if district_field:
                    dist_val = props.get(district_field, 'Unknown')
                    color = senate_colors.get(dist_val, '#87CEEB')
                else:
                    color = '#87CEEB'
                
                return {
                    'fillColor': color,
                    'color': '#1e40af',
                    'weight': 2.5,
                    'opacity': 0.9,
                    'fillOpacity': 0.45
                }
            
            tooltip_fields = [district_field] if district_field else []
            tooltip_aliases = ['Senate District:'] if district_field else []
            
            folium.GeoJson(
                data=geojson_obj,
                name="Senate Districts",
                style_function=senate_style_function,
                highlight_function=lambda x: {'weight': 4, 'fillOpacity': 0.7},
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="background-color: white; color: #111827; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);"
                ) if tooltip_fields else None
            ).add_to(m)
    
    if show_highways and 'highways_gdf' in map_data and not map_data['highways_gdf'].empty:
        highways_gdf = map_data['highways_gdf']
        geojson_obj = _gdf_to_geojson(highways_gdf)
        if geojson_obj and geojson_obj.get("features"):
            def highway_style_function(feature):
                return {
                    'color': '#666666',
                    'weight': 2,
                    'opacity': 0.7
                }
            
            folium.GeoJson(
                data=geojson_obj,
                name="Highways",
                style_function=highway_style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['FULLNAME'] if 'FULLNAME' in highways_gdf.columns else [],
                    aliases=['Highway:'] if 'FULLNAME' in highways_gdf.columns else [],
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="background-color: white; color: #111827; font-size: 14px; padding: 8px; border-radius: 5px;"
                ) if 'FULLNAME' in highways_gdf.columns else None
            ).add_to(m)
    
    icon_create_fn = (
        "function(cluster) {"
        "  var count = cluster.getChildCount();"
        "  return L.divIcon({"
        "    html: '<div style=\"background-color: #007A33; color: white; border-radius: 50%;"
        "                 width: 42px; height: 42px; display: flex; align-items: center;"
        "                 justify-content: center; font-weight: 700; font-size: 13px;\">' + count + '</div>',"
        "    className: 'custom-div-icon',"
        "    iconSize: [42, 42],"
        "    iconAnchor: [21, 21]"
        "  });"
        "}"
    )
    
    marker_cluster = MarkerCluster(
        name="Highway Projects",
        overlay=True,
        control=True,
        icon_create_function=icon_create_fn
    ).add_to(m)
    
    for _, row in filtered_df.iterrows():
        lat = row.get('latitude')
        lng = row.get('longitude')
        
        try:
            lat = float(lat) if pd.notnull(lat) else None
            lng = float(lng) if pd.notnull(lng) else None
        except:
            lat, lng = None, None
        
        if lat is not None and lng is not None:
            impact = int(row.get('Total_Relocations') or 0)
            if impact >= 100:
                mcolor = '#9B1C1C'
            elif impact >= 20:
                mcolor = '#D97706'
            elif impact > 0:
                mcolor = '#F59E0B'
            else:
                mcolor = '#007A33'
            
            proj_id = row.get('Project') or 'Unknown'
            location_text = row.get('Project_Location') or 'N/A'
            district_text = row.get('district_num') or 'N/A'
            county_text = row.get('County') or 'N/A'
            route_text = row.get('Route') or 'N/A'
            year_text = row.get('CCA_FY') or 'N/A'
            homes_dem = int(row.get('Num_Home_Demolished') or 0)
            bus_dem = int(row.get('Num_Business_Demolished') or 0)
            total_rel = int(row.get('Total_Relocations') or 0)
            
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 320px;">
                <h4 style="color:#007A33; margin-bottom: 6px;">Project {proj_id}</h4>
                <div style="font-size:13px; color:#111827;">
                    <p style="margin:0;"><strong>Location:</strong> {district_text}</p>
                    <p style="margin:0;"><strong>County:</strong> {county_text}</p>
                    <p style="margin:0;"><strong>Route:</strong> {route_text}</p>
                    <p style="margin:0;"><strong>Year:</strong> {year_text}</p>
                    <hr style="margin:8px 0;">
                    <p style="margin:0;"><strong>Impact Summary</strong></p>
                    <p style="margin:0;">🏠 Homes Demolished: {homes_dem}</p>
                    <p style="margin:0;">🏢 Businesses Demolished: {bus_dem}</p>
                    <p style="margin:0;">📦 Total Relocations: {total_rel}</p>
                </div>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lng],
                radius=8 if impact == 0 else min(18, 6 + int((impact ** 0.5) / 1.5)),
                color=mcolor,
                fill=True,
                fill_color=mcolor,
                fill_opacity=0.9,
                popup=folium.Popup(popup_content, max_width=360),
                tooltip=f"Project {proj_id} – {total_rel} displacements"
            ).add_to(marker_cluster)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    # DISPLAY MAP IN THE PLACEHOLDER AT THE TOP
    try:
        with map_placeholder.container():
            st_folium(m, width=None, height=800, returned_objects=[])
    except Exception as ex:
        st.error("Error rendering map.")
        logger.error(f"Map error: {ex}")
    
    st.session_state.map_rendered = True
    
    # DISPLAY DATA TABLE WITH FILTERED DATA
    st.markdown("### 📋 Project Details")
    
    display_columns = [
        'Project', 'County', 'Assemblymember 1', 'Assemblymember 2', 'Assemblymember 3', 
        'Senator 1', 'Senator 2', 'Route', 'CCA_FY', 'Project_Location',
        'Num_Home_Demolished', 'Num_Business_Demolished', 'Total_Relocations'
    ]
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    if available_columns and not filtered_df.empty:
        styled_df = filtered_df[available_columns].style.format({
            'Num_Home_Demolished': '{:.0f}',
            'Num_Business_Demolished': '{:.0f}',
            'Total_Relocations': '{:.0f}'
        }).background_gradient(
            subset=['Total_Relocations'], 
            cmap='Reds', 
            vmin=0, 
            vmax=filtered_df['Total_Relocations'].max() if filtered_df['Total_Relocations'].max() > 0 else 1
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)    
