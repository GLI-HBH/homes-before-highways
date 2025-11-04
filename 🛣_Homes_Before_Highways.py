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

# Set page configuration
st.set_page_config(
    page_title="Homes Before Highways", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🏠"
)

# Enhanced CSS for scrolling design
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: #F9FAF9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1C1C1C;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #007A33 0%, #005924 100%);
        color: white;
        padding: 4rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    /* Expandable sections */
    .section-container {
        background: #FFFFFF;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .section-container:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .section-header {
        background: linear-gradient(135deg, #007A33 0%, #065F46 100%);
        color: white;
        padding: 1.5rem 2rem;
        margin: 0;
        cursor: pointer;
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .section-header:hover {
        background: linear-gradient(135deg, #065F46 0%, #007A33 100%);
    }
    
    .section-content {
        padding: 2rem;
        display: none;
    }
    
    .section-content.expanded {
        display: block;
        animation: expandDown 0.3s ease-out;
    }
    
    @keyframes expandDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .expand-icon {
        font-size: 1.2rem;
        transition: transform 0.3s ease;
    }
    
    .expand-icon.rotated {
        transform: rotate(180deg);
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
    
    /* Image carousel */
    .carousel-container {
        text-align: center;
        background: #F8FAFC;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .carousel-image {
        max-width: 300px;
        max-height: 300px;
        border-radius: 50%;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        transition: all 0.5s ease;
    }
    
    .carousel-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #007A33;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Call to action */
    .cta-container {
        background: linear-gradient(135deg, #007A33 0%, #005924 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 12px 32px rgba(0,122,51,0.3);
    }
    
    .cta-button {
        background: white;
        color: #007A33;
        border: none;
        padding: 15px 30px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .cta-button:hover {
        background: #F3F6F4;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .hero-subtitle { font-size: 1.1rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .section-header { font-size: 1.3rem; padding: 1rem 1.5rem; }
        .section-content { padding: 1.5rem; }
    }
</style>

<script>
function toggleSection(sectionId) {
    const content = document.getElementById(sectionId + '-content');
    const icon = document.getElementById(sectionId + '-icon');
    
    if (content.classList.contains('expanded')) {
        content.classList.remove('expanded');
        icon.classList.remove('rotated');
        content.style.display = 'none';
    } else {
        content.classList.add('expanded');
        icon.classList.add('rotated');
        content.style.display = 'block';
    }
}
</script>
""", unsafe_allow_html=True)

# Header with logo
left_co, cent_co, right_co = st.columns(3)
with cent_co:
    st.image("GLI Logo.jpg")
    

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



# Load data
df = load_data()
map_data = load_map_data()

# Hero section
st.markdown("""
<div class="hero-container" style="text-align: center; margin-top: 2em; margin-bottom: 2em;">
    <h1 class="hero-title" style="font-size: 3em; margin-bottom: 0.5em; color: white;">🏠 Homes Before Highways</h1>
    <p class="hero-subtitle" style="font-size: 1.5em; line-height: 1.6; max-width: 800px; margin: 0 auto; color: white;">
        Tracking the true costs of highway expansion and building a future that protects homes, communities, and everyday Californians
    </p>
</div>
""", unsafe_allow_html=True)


# ===== OVERVIEW SECTION =====
st.markdown("""
<div class="section-container">
    <div class="section-header" onclick="toggleSection('overview')">
        <span>🏠 Overview: The Crisis in Numbers</span>
        <span id="overview-icon" class="expand-icon">▼</span>
    </div>
    <div id="overview-content" class="section-content">
""", unsafe_allow_html=True)

# Calculate key stats
if not df.empty:
    total_homes = int(df["Num_Home_Demolished"].sum())
    total_businesses = int(df["Num_Business_Demolished"].sum())
    total_relocations = int(df["Total_Relocations"].sum())
    total_projects = len(df)
else:
    total_homes = total_businesses = total_relocations = total_projects = 0

# Enhanced metrics display
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{total_homes}</div>
        <div class="metric-label">Homes Demolished<br>(2018-2023)</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_businesses}</div>
        <div class="metric-label">Businesses Destroyed<br>(2018-2023)</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">550+</div>
        <div class="metric-label">Lane Miles Added<br>(Length of California)</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">200+</div>
        <div class="metric-label">Planned Highway<br>Expansions</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Problem statement
st.markdown("""
## 🚧 Highway Expansion is Worsening the Housing Crisis

For decades, highway construction has divided neighborhoods, displaced families, 
shuttered businesses and polluted air, primarily in low-income communities and 
communities of color. **This destruction is still happening today.**

With more than 200 planned highway expansions just in California, many more 
families and communities remain at risk. These are not isolated incidents—it's 
a systematic failure in transportation planning that worsens traffic while failing 
to deliver promised benefits.
""")

# Cost analysis
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ## 💰 High Costs, Low Return
    
    California has the largest transportation budget in the nation. **In 2024 alone, 
    the state spent $30.4 billion** on transportation. Despite this massive spending:
    
    - **California roads rank 47th** in quality nationwide
    - **Californians pay up to $1,000/year** in vehicle maintenance due to poor roads
    - **Drivers lose 88 hours/year** stuck in traffic in Los Angeles alone
    - **California is the 3rd deadliest state** for pedestrians
    
    **The result:** Enormous costs to our homes, health, and communities, with none of the promised gains.
    """)

with col2:
    # Create a simple cost visualization
    cost_data = {
        'Category': ['Vehicle\nMaintenance', 'Extra Gas\nCosts', 'Time Lost\n(Value)', 'Health\nImpacts'],
        'Annual Cost per Person': [1000, 800, 2200, 1500]
    }
    
    fig = px.bar(cost_data, x='Category', y='Annual Cost per Person',
                title="Annual Cost of Poor Transportation Policy",
                color='Annual Cost per Person',
                color_continuous_scale='Reds')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#1e3a8a',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)


# Load the image
img2 = Image.open("Los Angeles Interstate 5.png")

# Convert to base64
buffered = BytesIO()
img2.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Display centered with width 600
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_str}" width="800">
    </div>
    """,
    unsafe_allow_html=True
)

# ===== INTERACTIVE MAP SECTION =====
st.markdown("""
<div class="section-container">
    <div class="section-header" onclick="toggleSection('map')">
        <span>🗺️ Interactive Map</span>
        <span id="map-icon" class="expand-icon">▼</span>
    </div>
    <div id="map-content" class="section-content">
""", unsafe_allow_html=True)

st.markdown("### 🔍 Explore Highway Project Impacts")

# Enhanced filter controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    if not df.empty:
        counties = ["All Counties"] + sorted(df["County"].dropna().unique().tolist())
        selected_county = st.selectbox("📍 Filter by County", counties)
    else:
        selected_county = "All Counties"

with col2:
    if not df.empty:
        routes = ["All Routes"] + sorted(df["Route"].dropna().astype(str).unique().tolist(), key=lambda x: (not x.isdigit(), x))
        selected_route = st.selectbox("🛣️ Filter by Route", routes)
    else:
        selected_route = "All Routes"

with col3:
    if not df.empty:
        years = ["All Years"] + sorted(df["CCA_FY"].dropna().astype(str).unique().tolist())
        selected_year = st.selectbox("📅 Filter by Year", years)
    else:
        selected_year = "All Years"

with col4:
    impact_filter = st.selectbox("🎯 Impact Level",
                                ["All Projects", "High Impact (100+)",
                                 "Medium Impact (20-100)", "Low Impact (1-20)", "No Impact"])

# Map layer controls
st.markdown("#### 🎨 Map Layers")
layer_cols = st.columns(4)

with layer_cols[0]:
    show_highways = st.checkbox("🛣️ CA Highways", value=True)
with layer_cols[1]:
    show_assembly = st.checkbox("🏛️ Assembly Districts", value=True)
with layer_cols[2]:
    show_senate = st.checkbox("🏛️ Senate Districts", value=True)
with layer_cols[3]:
    show_calenviro = st.checkbox("🌍 Environmental Justice", value=False)

# Apply filters and create map
if not df.empty:
    filtered_df = df.copy()
    
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
    
    # Apply impact filter
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
    
    # Create map
    if not filtered_df[filtered_df['latitude'].notnull()].empty:
        center_lat = float(filtered_df.loc[filtered_df['latitude'].notnull(), 'latitude'].mean())
        center_lng = float(filtered_df.loc[filtered_df['longitude'].notnull(), 'longitude'].mean())
        zoom_start = 7
    else:
        center_lat, center_lng, zoom_start = 37.2, -119.5, 6
    
    # Enhanced folium map with better styling
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles='CartoDB positron',
        prefer_canvas=True
    )
    
    # Helper function to safely convert GeoDataFrame to GeoJSON
    def _gdf_to_geojson(gdf):
        try:
            if hasattr(gdf, "to_json"):
                return json.loads(gdf.to_json())
            elif isinstance(gdf, dict):
                return gdf
        except Exception:
            return None
    
    # Add selected layers (safe checks + conversion)
    if show_highways and 'highways_gdf' in map_data and not map_data['highways_gdf'].empty:
        highways_gdf = map_data['highways_gdf']
        geojson_obj = _gdf_to_geojson(highways_gdf)
        if geojson_obj and geojson_obj.get("features"):
            tooltip_fields = []
            tooltip_aliases = []
            if 'FULLNAME' in highways_gdf.columns:
                tooltip_fields = ['FULLNAME']
                tooltip_aliases = ['Highway:']
            
            folium.GeoJson(
                data=geojson_obj,
                name="CA Highways",
                style_function=lambda feat: {
                    'color': '#4B5563',  # muted slate
                    'weight': 2,
                    'opacity': 0.8
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="background-color: white; color: #111827; font-size: 12px; padding: 6px;"
                ) if tooltip_fields else None
            ).add_to(m)
    
    # Add CalEnviroScreen layer
    if show_calenviro and 'calenviro_gdf' in map_data and not map_data['calenviro_gdf'].empty:
        calenviro_gdf = map_data['calenviro_gdf']
        geojson_obj = _gdf_to_geojson(calenviro_gdf)
        if geojson_obj and geojson_obj.get("features"):
            def ces_style_function(feature):
                props = feature.get('properties', {}) or {}
                score = props.get('CES_SCORE', None)
                try:
                    score = float(score)
                except Exception:
                    score = None
                # Use a green-tinged palette but still indicate burden
                if score is None:
                    fill = '#E6F4EC'
                elif score < 25:
                    fill = '#E6F4EC'   # light green
                elif score < 50:
                    fill = '#C7E7D0'
                elif score < 75:
                    fill = '#7FC08A'
                else:
                    fill = '#007A33'
                return {
                    'fillColor': fill,
                    'color': '#1F2937',
                    'weight': 0.5,
                    'fillOpacity': 0.6
                }
            
            tooltip_fields = []
            tooltip_aliases = []
            if 'CES_SCORE' in calenviro_gdf.columns:
                tooltip_fields.append('CES_SCORE'); tooltip_aliases.append('Environmental Burden Score:')
            if 'TRACT' in calenviro_gdf.columns:
                tooltip_fields.append('TRACT'); tooltip_aliases.append('Census Tract:')
            
            folium.GeoJson(
                data=geojson_obj,
                name="Environmental Justice Scores",
                style_function=ces_style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="background-color: white; color: #111827; font-size: 12px; padding: 6px;"
                ) if tooltip_fields else None
            ).add_to(m)
    
    # Add district boundaries
    if show_assembly and 'assembly_gdf' in map_data and not map_data['assembly_gdf'].empty:
        assembly_gdf = map_data['assembly_gdf']
        geojson_obj = _gdf_to_geojson(assembly_gdf)
        if geojson_obj and geojson_obj.get("features"):
            folium.GeoJson(
                data=geojson_obj,
                name="Assembly Districts",
                style_function=lambda feat: {
                    'fillColor': 'transparent',
                    'color': '#007A33',
                    'weight': 2,
                    'opacity': 0.8
                }
            ).add_to(m)
    
    if show_senate and 'senate_gdf' in map_data and not map_data['senate_gdf'].empty:
        senate_gdf = map_data['senate_gdf']
        geojson_obj = _gdf_to_geojson(senate_gdf)
        if geojson_obj and geojson_obj.get("features"):
            folium.GeoJson(
                data=geojson_obj,
                name="Senate Districts",
                style_function=lambda feat: {
                    'fillColor': 'transparent',
                    'color': '#065F46',
                    'weight': 2,
                    'opacity': 0.8
                }
            ).add_to(m)
    
    # Add project markers with enhanced clustering (brand green cluster)
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
    
    valid_locations = 0
    for _, row in filtered_df.iterrows():
        lat = row.get('latitude')
        lng = row.get('longitude')
        
        try:
            lat = float(lat) if pd.notnull(lat) else None
            lng = float(lng) if pd.notnull(lng) else None
        except:
            lat, lng = None, None
        
        if lat is not None and lng is not None:
            valid_locations += 1
            
            impact = int(row.get('Total_Relocations') or 0)
            if impact >= 100:
                mcolor = '#9B1C1C'   # deep red for very high impact
            elif impact >= 20:
                mcolor = '#D97706'   # amber
            elif impact > 0:
                mcolor = '#F59E0B'   # yellow-amber
            else:
                mcolor = '#007A33'   # green for no impact
            
            proj_id = row.get('Project') or row.get('Project_ID') or row.get('PROJECT_ID') or 'Unknown'
            location_text = row.get('Project_Location') or row.get('Location') or 'N/A'
            county_text = row.get('County') or 'N/A'
            route_text = row.get('Route') or 'N/A'
            year_text = row.get('CCA_FY') or row.get('Year') or 'N/A'
            homes_dem = int(row.get('Num_Home_Demolished') or 0)
            bus_dem = int(row.get('Num_Business_Demolished') or 0)
            total_rel = int(row.get('Total_Relocations') or 0)
            
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 320px;">
                <h4 style="color:#007A33; margin-bottom: 6px;">Project {proj_id}</h4>
                <div style="font-size:13px; color:#111827;">
                    <p style="margin:0;"><strong>Location:</strong> {location_text}</p>
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
            
            # Use a CircleMarker for consistent styling
            folium.CircleMarker(
                location=[lat, lng],
                radius=8 if impact == 0 else min(18, 6 + int((impact ** 0.5) / 1.5)),
                color=mcolor,
                fill=True,
                fill_color=mcolor,
                fill_opacity=0.9,
                popup=folium.Popup(popup_content, max_width=360),
                tooltip=f"Project {proj_id} — {total_rel} displacements"
            ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Display map with error handling
    try:
        map_data_result = st_folium(m, width=1200, height=700, returned_objects=["last_object_clicked"])
    except AssertionError as e:
        st.error("Map rendering encountered an issue (likely GeoJSON/data mismatch). Showing fallback map view.")
        try:
            # Fallback: create a simple map without the problematic layers
            simple_m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=zoom_start,
                tiles='CartoDB positron'
            )
            # Add only the project markers
            for _, row in filtered_df.iterrows():
                lat = row.get('latitude')
                lng = row.get('longitude')
                if pd.notnull(lat) and pd.notnull(lng):
                    folium.CircleMarker(
                        location=[float(lat), float(lng)],
                        radius=8,
                        color='#007A33',
                        fill=True,
                        popup=f"Project {row.get('Project', 'Unknown')}"
                    ).add_to(simple_m)
            st_folium(simple_m, width=1200, height=600)
        except Exception:
            st.error("Unable to render map. Please check data files.")
    except Exception as ex:
        st.error("Unexpected error rendering map.")
        logger.error(f"Map error: {ex}")
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projects Displayed", valid_locations)
    with col2:
        if valid_locations > 0:
            avg_impact = filtered_df['Total_Relocations'].dropna()
            if len(avg_impact) > 0:
                st.metric("Average Impact", f"{avg_impact.mean():.1f} displacements")
            else:
                st.metric("Average Impact", "—")
    with col3:
        total_impact = int(filtered_df['Total_Relocations'].fillna(0).sum())
        st.metric("Total Impact", f"{total_impact} displacements")

else:
    st.info("No data available for mapping. Please check that the data file is loaded correctly.")

st.markdown("</div></div>", unsafe_allow_html=True)


# ===== DATA DASHBOARD SECTION =====
st.markdown("""
<div class="section-container">
    <div class="section-header" onclick="toggleSection('dashboard')">
        <span>📊 Data Dashboard</span>
        <span id="dashboard-icon" class="expand-icon">▼</span>
    </div>
    <div id="dashboard-content" class="section-content">
""", unsafe_allow_html=True)

st.markdown("### 🔍 Search & Filter Projects")

# Enhanced search and filtering
col1, col2, col3, col4 = st.columns(4)

with col1:
    search_text = st.text_input("🔎 Search Projects", placeholder="Enter Project or location...")

with col2:
    if not df.empty:
        counties = ["All Counties"] + sorted(df["County"].dropna().unique().tolist())
        filter_county = st.selectbox("📍 County Filter", counties, key="dash_county")
    else:
        filter_county = "All Counties"

with col3:
    if not df.empty:
        routes = ["All Routes"] + sorted([str(r) for r in df["Route"].dropna().unique()])
        filter_route = st.selectbox("🛣️ Route Filter", routes, key="dash_route")
    else:
        filter_route = "All Routes"

with col4:
    if not df.empty:
        years = ["All Years"] + sorted([str(y) for y in df["CCA_FY"].dropna().unique()])
        filter_year = st.selectbox("📅 Year Filter", years, key="dash_year")
    else:
        filter_year = "All Years"

# Apply filters
if not df.empty:
    display_df = df.copy()
    
    if search_text:
        search_mask = display_df.apply(
            lambda row: any(
                str(search_text).lower() in str(val).lower() 
                for val in row.values if pd.notnull(val)
            ), axis=1
        )
        display_df = display_df[search_mask]
    
    if filter_county != "All Counties":
        display_df = display_df[display_df["County"] == filter_county]
    
    if filter_route != "All Routes":
        try:
            display_df = display_df[display_df["Route"] == int(filter_route)]
        except:
            display_df = display_df[display_df["Route"].astype(str) == filter_route]
    
    if filter_year != "All Years":
        try:
            display_df = display_df[display_df["CCA_FY"] == int(filter_year)]
        except:
            display_df = display_df[display_df["CCA_FY"].astype(str) == filter_year]
    
    # Project summary metrics
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{len(display_df)}</div>
            <div class="metric-label">Filtered Projects</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{int(display_df['Num_Home_Demolished'].sum())}</div>
            <div class="metric-label">Homes Demolished</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{int(display_df['Num_Business_Demolished'].sum())}</div>
            <div class="metric-label">Businesses Demolished</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{int(display_df['Total_Relocations'].sum())}</div>
            <div class="metric-label">Total Relocations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced data table
    st.markdown("### 📋 Project Details")
    
    # Select columns for display
    display_columns = [
        'Project', 'County', 'Route', 'CCA_FY', 'Project_Location',
        'Num_Home_Demolished', 'Num_Business_Demolished', 'Total_Relocations'
    ]
    available_columns = [col for col in display_columns if col in display_df.columns]
    
    # Enhanced dataframe with styling
    if available_columns and not display_df.empty:
        styled_df = display_df[available_columns].style.format({
            'Num_Home_Demolished': '{:.0f}',
            'Num_Business_Demolished': '{:.0f}',
            'Total_Relocations': '{:.0f}'
        }).background_gradient(
            subset=['Total_Relocations'], 
            cmap='Reds', 
            vmin=0, 
            vmax=display_df['Total_Relocations'].max() if display_df['Total_Relocations'].max() > 0 else 1
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download functionality
        col1, col2 = st.columns(2)
        with col1:
            csv = display_df[available_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"highway_projects_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export to CSV instead of Excel to avoid openpyxl dependency
            csv_buffer = io.StringIO()
            display_df[available_columns].to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download as CSV (Alternative)",
                data=csv_buffer.getvalue(),
                file_name=f"highway_projects_{datetime.now().strftime('%Y%m%d')}_alt.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No project data available to display.")

else:
    st.info("No data available. Please check that the data file is loaded correctly.")

st.markdown("</div></div>", unsafe_allow_html=True)

# ===== ANALYSIS SECTION =====
st.markdown("""
<div class="section-container">
    <div class="section-header" onclick="toggleSection('analysis')">
        <span>📈 Analysis & Insights</span>
        <span id="analysis-icon" class="expand-icon">▼</span>
    </div>
    <div id="analysis-content" class="section-content">
""", unsafe_allow_html=True)

st.markdown("### 📊 Comprehensive Data Analysis")

if not df.empty:
    # County analysis
    st.markdown("#### 🏛️ County-Level Impact Analysis")
    
    county_analysis = df.groupby('County').agg({
        'Project': 'count',
        'Num_Home_Demolished': 'sum',
        'Num_Business_Demolished': 'sum',
        'Total_Relocations': 'sum'
    }).reset_index()
    
    county_analysis.columns = ['County', 'Projects', 'Homes_Demolished', 
                              'Businesses_Demolished', 'Total_Relocations']
    county_analysis = county_analysis.sort_values('Total_Relocations', ascending=False)
    
    # Top 10 counties chart
    top_counties = county_analysis.head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(top_counties, x='County', y='Total_Relocations',
                    title="Top 10 Counties by Total Relocations",
                    color='Total_Relocations', color_continuous_scale='Reds')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(county_analysis, x='Projects', y='Total_Relocations', 
                        size='Total_Relocations', hover_data=['County'],
                        title="Projects vs. Impact by County",
                        color='Total_Relocations', color_continuous_scale='Reds')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.markdown("#### 📅 Temporal Analysis")
    
    yearly_analysis = df.groupby('CCA_FY').agg({
        'Project': 'count',
        'Total_Relocations': 'sum'
    }).reset_index()
    yearly_analysis.columns = ['Year', 'Projects', 'Total_Relocations']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(yearly_analysis, x='Year', y='Projects', 
                     title='Projects by Year',
                     markers=True, line_shape='spline')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(yearly_analysis, x='Year', y='Total_Relocations',
                    title='Relocations by Year',
                    color='Total_Relocations', color_continuous_scale='Reds')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Route analysis
    st.markdown("#### 🛣️ Highway Route Analysis")
    
    route_analysis = df.groupby('Route').agg({
        'Project': 'count',
        'Total_Relocations': 'sum'
    }).reset_index()
    route_analysis.columns = ['Route', 'Projects', 'Total_Relocations']
    route_analysis = route_analysis.sort_values('Total_Relocations', ascending=False).head(15)
    
    fig = px.scatter(route_analysis, x='Projects', y='Total_Relocations', 
                    size='Total_Relocations', hover_data=['Route'],
                    title='Projects vs. Impact by Highway Route',
                    color='Total_Relocations', color_continuous_scale='Reds')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No data available for analysis.")

st.markdown("</div></div>", unsafe_allow_html=True)

# ===== CASE STUDIES SECTION =====
st.markdown("""
<div class="section-container">
    <div class="section-header" onclick="toggleSection('cases')">
        <span>📚 Case Studies</span>
        <span id="cases-icon" class="expand-icon">▼</span>
    </div>
    <div id="cases-content" class="section-content">
""", unsafe_allow_html=True)

st.markdown("### 📚 Case Studies: Real Communities, Real Impact")

# Los Angeles I-5 Case Study
st.markdown("#### 🚨 Los Angeles - Interstate 5: Most Destructive Expansion")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Between Burbank and Norwalk, Caltrans demolished **569 homes and businesses** 
    along the corridor, accounting for over 90% of all demolitions statewide for 
    highway expansions completed since 2018.
    
    **Key Community Impacts:**
    - 💰 $1.3 billion project delayed five years, required $73M loan from LA Metro
    - 🏠 96% of nearby households are housing-burdened
    - 🌍 Community ranks in 98th percentile of CalEnviroScreen (worse environmental 
      burdens than 98% of the state)
    - 👨‍👩‍👧‍👦 300 families displaced from their homes
    """)

with col2:
    # Create a simple impact visualization for I-5
    impact_data = pd.DataFrame({
        'Impact Type': ['Homes', 'Businesses', 'Total Cost (Millions)'],
        'Value': [300, 269, 1300]
    })
    
    fig = px.bar(impact_data, x='Impact Type', y='Value',
                title="I-5 Project Impact",
                color='Value', color_continuous_scale='Reds')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# Other case studies
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🍪 Fresno - State Route 99: Small Businesses Displaced
    
    Caltrans spent $146 million expanding State Route 99, displacing 11 businesses and increasing heavy truck traffic.
    
    **Key Community Impacts:**
    - Loss of 11 local businesses disrupted the local economy and neighborhood stability.
    - 72% of nearby households are housing-burdened (spending >50% of income on housing).
    - The area ranks in the 90th—99th percentile on CalEnviroScreen for combined burdens.
    """)

with col2:
    st.markdown("""
    #### 🛣️ Los Angeles - Interstate 405: Billions Spent, Traffic Worsened
    
    The $2.16 billion expansion of I-405 demolished dozens of homes and businesses,
    and early post-construction data indicate commute times did not improve as promised.
    
    **Key Community Impacts:**
    - $2.16 billion spent — project costs ballooned while benefits lagged.
    - 20 homes and 3 businesses demolished between 2018—2023.
    - Nearby neighborhoods show elevated pollution and traffic-related health risks.
    """)

st.markdown("</div></div>", unsafe_allow_html=True)

# ===== TEAM SECTION =====
# Centering CSS
st.markdown("""
<style>
.section-container {
    text-align: center;
}
.section-header {
    cursor: pointer;
    display: inline-block;  /* Keeps click area tight around the text */
}
.section-content {
    text-align: center;
}
h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load the image
img3 = Image.open("Small Businesses Displaced.png")

# Convert to base64
buffered = BytesIO()
img3.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Display centered with width 600
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_str}" width="600">
    </div>
    """,
    unsafe_allow_html=True
)


# ===== TAKE ACTION SECTION =====
st.markdown("""
<div class="section-container">
    <div class="section-header" onclick="toggleSection('action')">
        <span>🎯 Take Action</span>
        <span id="action-icon" class="expand-icon">▼</span>
    </div>
    <div id="action-content" class="section-content">
""", unsafe_allow_html=True)

# Hero call to action
st.markdown("""
<div class="cta-container">
    <h2>🌟 A Better Way Forward</h2>
    <p>California doesn't have to choose between efficient transportation, affordability, and healthy communities. 
       We can build a system that delivers all three by investing in strategies that:</p>
    <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
        <li>✅ Expand affordable, climate-friendly public transit</li>
        <li>🏠 Protect communities from displacement and preserve affordable housing</li>
        <li>🌱 Cut pollution by investing in EV infrastructure</li>
        <li>💼 Create good local jobs through community-strengthening investments</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Action items
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📝 Sign the Petition
    
    Add your voice to demand that California prioritize homes before highways. 
    Join thousands of Californians calling for transportation justice.
    """)
    
    if st.button("Sign Now", key="petition", use_container_width=True):
        st.success("Thank you for your interest! In the full application, this would link to the petition.")

with col2:
    st.markdown("""
    ### 📢 Share the Map
    
    Help spread awareness by sharing this interactive map with your community, 
    local officials, and social networks.
    """)
    
    if st.button("Share Map", key="share", use_container_width=True):
        st.success("Sharing functionality would be implemented in the full application.")

with col3:
    st.markdown("""
    ### 🤝 Join the Coalition
    
    Connect with local organizations and advocates working for transportation 
    equity in your community.
    """)
    
    if st.button("Get Involved", key="coalition", use_container_width=True):
        st.success("Coalition information would be provided in the full application.")

# Final takeaways
st.markdown("""
### 📚 Key Takeaways

- Highway expansions are not a neutral infrastructure choice — they have measurable,
  unequal harms on housing stability, small businesses, and public health.
- The evidence suggests widening highways often produces worse traffic outcomes, higher
  costs, and substantial community disruption.
- We can prioritize alternatives: transit, preservation of housing, EV infrastructure,
  and community-driven investments that create local jobs.
""")

st.markdown("</div></div>", unsafe_allow_html=True)


# Social Media Icons
from st_social_media_links import SocialMediaIcons
social_media_links = [
    "https://www.linkedin.com/company/thegreenlininginstitute-/",
    "https://www.instagram.com/greenlining/",
    "https://www.facebook.com/Greenlining/",
    "https://www.youtube.com/channel/UCKxMsA3yBiLiz_3g-dTiFIg",
    "https://twitter.com/greenlining/",
    "https://www.tiktok.com/@greenlining",
]

social_media_icons = SocialMediaIcons(social_media_links)
# social_media_icons.render(sidebar=True, justify_content="center")

st.divider()
st.write("")

social_media_icons = SocialMediaIcons(social_media_links) 
social_media_icons.render(sidebar=False, justify_content="center")


# Footer
st.markdown("""
<div style="text-align:center; margin-top: 3rem; padding: 2rem; background: #F3F6F4; border-radius: 16px;">
    <div style="color: #007A33; font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem;">
        Built by The Greenlining Institute — Homes Before Highways Project
    </div>
    <div style="color: #6b7280; margin-bottom: 1rem;">
        Data sources: SB 695 (Select State Highway System Project Outcomes), Caltrans project records, CalEnviroScreen
    </div>
    <div style="color: #9ca3af; font-size: 0.9rem;">
        This tool is for advocacy and informational use. For errors or corrections, contact marc.guirand@greenlining.org
    </div>
</div>
""", unsafe_allow_html=True)

# JavaScript for section toggling (since Streamlit doesn't execute the inline onclick)
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Make sections clickable
    const headers = document.querySelectorAll('.section-header');
    headers.forEach(header => {
        header.addEventListener('click', function() {
            const sectionId = this.onclick.toString().match(/'(\w+)'/)[1];
            toggleSection(sectionId);
        });
    });
});
</script>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Engagement / contact panel
st.markdown("""
<div class="card" style="text-align: center; margin: 2em 0;">
    <h3 style="margin-bottom: 0.5em;">📬 Get In Touch / Feedback</h3>
    <p style="font-size: 1.1em; line-height: 1.6;">
        Questions, corrections, or stories to share from your community? We want to hear from you.<br>
        Please email: <strong> Hana Creger (hana.creger@greenlining.org)</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Markdown header centered
st.markdown("### 👥 The Greenlining Institute Team", unsafe_allow_html=True)

# Predefined list of image names
DEFAULT_IMAGES = [
    "Hana Creger.png", 
    "Yesenia Perez.png",
    "Gyasi Pigott.png",
    "Dohee Kim.png",	
    "Marc H. Guirand.png"
]
    
def get_images(image_list=None):
    """
    Load images from the current directory.
    
    Args:
        image_list (list): Optional list of image names to load. If None, uses DEFAULT_IMAGES.
    
    Returns:
        list: Loaded PIL Image objects with their names.
    """
    # Use default images if no list is provided
    if image_list is None:
        image_list = DEFAULT_IMAGES
    
    # Load images
    images = []
    for img_name in image_list:
        # Check if the file exists
        if not os.path.exists(img_name):
            st.warning(f"Image not found: {img_name}")
            continue
        
        try:
            # Try opening the image
            img = Image.open(img_name)
            # Store image and its name (without file extension)
            images.append({
                'image': img, 
                'name': os.path.splitext(img_name)[0]
            })
        except UnidentifiedImageError:
            st.error(f"File is not a valid image: {img_name}")
        except Exception as e:
            st.error(f"Error loading image {img_name}: {e}")
    
    return images

def auto_scroll_gallery():
    """Create a Streamlit app with auto-scrolling image gallery."""
    # Center-align content using a single centered column
    col_center = st.columns([1, 6, 1])[1]
    
    with col_center:
        
        # Load images
        images = get_images()
        
        # Only show gallery if images are available
        if images:
            # Create placeholders for image display and subtitle
            image_placeholder = st.empty()
            subtitle_placeholder = st.empty()
            
            # Auto-scroll logic
            while True:
                for img_data in images:
                    # Get image dimensions
                    img_width, img_height = img_data['image'].size
                    
                    # Render image with custom HTML for centering
                    image_placeholder.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{image_to_base64(img_data['image'])}" 
                                 width="{img_width}" height="{img_height}" style="margin: auto;"/>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    # Display subtitle (name of the image) and center it
                    subtitle_placeholder.markdown(
                        f"<h3 style='text-align: center;'>{img_data['name']}</h3>", 
                        unsafe_allow_html=True
                    )
                    
                    # Wait for 2 seconds before showing the next image
                    time.sleep(2)
        else:
            st.info("Please add valid images to the current directory.")

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    import io
    import base64
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return img_base64

def main():
    # Run the auto-scroll gallery
    auto_scroll_gallery()

def main():
    auto_scroll_gallery()

st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()


