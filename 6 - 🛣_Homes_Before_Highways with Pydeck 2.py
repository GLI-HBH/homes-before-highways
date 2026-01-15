# Standard libraries
import os
import json
import logging

# Data handling
import pandas as pd
import numpy as np

# Visualization
import pydeck as pdk

# Streamlit
import streamlit as st


# Safety defaults (prevents NameError if a UI block is skipped)
show_projects = True  # safety default
# Set page configuration - MAKE WIDE BY DEFAULT
st.set_page_config(
    page_title="Homes Before Highways | The Greenlining Institute", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🏠"
)

# Enhanced CSS with Greenlining Institute branding
st.markdown("""
<style>
    /* Greenlining Institute Brand Colors */
    :root {
        --greenlining-primary: #00A651;
        --greenlining-dark: #007A33;
        --greenlining-light: #4CBB87;
        --greenlining-accent: #FFB81C;
        --greenlining-gray: #54565B;
        --greenlining-bg: #F8FBF9;
    }
    
    /* Global Styles */
    /* Make app feel fullscreen */
    div.block-container {max-width: 100% !important; padding-top: 0.75rem; padding-left: 1rem; padding-right: 1rem;}
    header, footer {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #F8FBF9 0%, #FFFFFF 100%);
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1C1C1C;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Hero section - Greenlining Style */
    .hero-container {
        background: linear-gradient(135deg, #00A651 0%, #007A33 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 166, 81, 0.3);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
        z-index: 0;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.75rem;
        text-shadow: 0 4px 12px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 1rem;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.7;
        position: relative;
        z-index: 1;
        opacity: 0.95;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        z-index: 1;
    }
    
    /* Filter section */
    .filter-container {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 166, 81, 0.08);
        border: 1px solid rgba(0, 166, 81, 0.1);
    }
    
    .filter-title {
        color: #00A651;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-title {
        color: #007A33;
        font-size: 1.75rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Map container */
    .map-container {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 12px 48px rgba(0, 166, 81, 0.12);
        border: 2px solid rgba(0, 166, 81, 0.15);
    }
    
    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FBF5 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00A651, #4CBB87);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .metric-card:hover {
        border: 2px solid #00A651;
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(0, 166, 81, 0.2);
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00A651 0%, #007A33 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 8px rgba(0, 166, 81, 0.1);
    }
    
    .metric-label {
        font-size: 1rem;
        color: #54565B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-sublabel {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.25rem;
    }
    
    /* Layer controls */
    .layer-controls {
        background: linear-gradient(135deg, #F8FBF9 0%, #FFFFFF 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(0, 166, 81, 0.1);
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Buttons and inputs with Greenlining colors */
    .stButton>button {
        background: linear-gradient(135deg, #00A651 0%, #007A33 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 166, 81, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 166, 81, 0.4);
    }
    
    /* Checkbox styling */
    .stCheckbox {
        color: #007A33;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
        border-left: 5px solid #00A651;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 166, 81, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFFDE7 100%);
        border-left: 5px solid #FFB81C;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-subtitle { font-size: 1rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F8FBF9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00A651 0%, #007A33 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #007A33;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data loading with performance optimization
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        # Faster CSV parsing when available (pyarrow engine)
        try:
            df = pd.read_csv("highway_projects_with_districts.csv", engine="pyarrow")
        except Exception:
            df = pd.read_csv("highway_projects_with_districts.csv")
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Num_Home_Demolished', 'Num_Business_Demolished', 'Total_Relocations']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Extract coordinates (vectorized for speed)
        if 'Project Start Point' in df.columns and 'Project End Point' in df.columns:
            # Accept formats like: "34.05,-118.25" or '34.05, -118.25' with optional quotes/spaces
            coord_re = r'\s*"?\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*"?\s*'
            start = df['Project Start Point'].astype(str).str.extract(coord_re)
            end = df['Project End Point'].astype(str).str.extract(coord_re)

            df['start_lat'] = pd.to_numeric(start[0], errors='coerce')
            df['start_lng'] = pd.to_numeric(start[1], errors='coerce')
            df['end_lat'] = pd.to_numeric(end[0], errors='coerce')
            df['end_lng'] = pd.to_numeric(end[1], errors='coerce')

            # Calculate project centers
            df['latitude'] = df[['start_lat', 'end_lat']].mean(axis=1)
            df['longitude'] = df[['start_lng', 'end_lng']].mean(axis=1)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load GeoJSON data without GeoPandas (Snowflake compatible)
@st.cache_data(ttl=7200, show_spinner=False)
def load_geojson_data():
    """Load GeoJSON files directly"""
    geojson_data = {}

    try:
        paths = {
            "assembly": "California Assembly Districts.geojson",
            "senate": "California Senate Districts.geojson",
            "highways": "california_highways.geojson",
        }

        for key, path in paths.items():
            if os.path.exists(path):
                with open(path, "r") as f:
                    geojson_data[key] = json.load(f)

        # Pre-compute tooltips + per-district colors (keeps render fast)
        # Requested: Assembly = light blue, Senate = light green
        if "assembly" in geojson_data:
            geojson_data["assembly"] = add_district_colors(
                geojson_data["assembly"],
                "Assembly District",
                base_hue="blue",
            )
        if "senate" in geojson_data:
            geojson_data["senate"] = add_district_colors(
                geojson_data["senate"],
                "Senate District",
                base_hue="green",
            )
        if "highways" in geojson_data:
            geojson_data["highways"] = add_highway_tooltips(geojson_data["highways"])


        return geojson_data

    except Exception as e:
        logger.error(f"Error loading GeoJSON: {e}")
        return {}

def _district_number_from_props(props: dict) -> str:
    """Best-effort district number extraction across common field names."""
    for key in ["DISTRICT", "District", "district", "SLDLST", "SD", "SLDUST", "GEOID", "GEOID10"]:
        val = props.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            return s
    return ""

def add_district_tooltip(geojson: dict, label_prefix: str) -> dict:
    """Inject a single tooltip_html field into each feature."""
    if not geojson or "features" not in geojson:
        return geojson

    for feat in geojson["features"]:
        props = feat.setdefault("properties", {})
        num = _district_number_from_props(props)
        # This is the ONLY thing that should appear when hovering districts
        props["tooltip_html"] = f"<b>{label_prefix}: {num}</b>" if num else f"<b>{label_prefix}</b>"

    return geojson

def _highway_name_from_props(props: dict) -> str:
    """Best-effort highway/route name extraction across common field names."""
    for key in [
        "Route", "ROUTE", "ROUTE_NUM", "RTE", "RT", "HWY", "HIGHWAY",
        "SR", "US", "I", "NAME", "FULLNAME", "FullName", "FULL_NAME", "ROAD_NAME"
    ]:
        val = props.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s and s.lower() != "nan":
            return s
    return ""

def add_highway_tooltips(geojson: dict) -> dict:
    """Inject tooltip_html into each highway feature so hover shows the highway name."""
    if not geojson or "features" not in geojson:
        return geojson

    for feat in geojson["features"]:
        props = feat.setdefault("properties", {})
        name = _highway_name_from_props(props)

        # Normalize numeric routes into a friendly label when possible
        label = ""
        if name:
            # If it's purely numeric (e.g., '880'), label as 'Route 880'
            if str(name).strip().isdigit():
                label = f"Route {str(name).strip()}"
            else:
                label = str(name).strip()
        else:
            # Fallback: try district-style keys or generic label
            label = "Highway Segment"

        props["tooltip_html"] = f"<b>{label}</b>"
    return geojson


# Generate distinct colors for districts (base_hue='green' requested)
def get_district_color(district_num, total_districts, base_hue: str = "green") -> str:
    """Generate distinct colors for districts as hex."""
    # Light, distinct palettes per user request:
    # - Senate: light green
    # - Assembly: light blue
    if base_hue == "green":
        # Greens (avoid super-yellow): ~115–150°
        hues = np.linspace(110, 155, total_districts)
    else:
        # Blues: ~195–225°
        hues = np.linspace(190, 235, total_districts)

    idx = int(district_num) % len(hues) if str(district_num).isdigit() else 0
    h = hues[idx]
    # Darker, more distinct fills (higher S, lower L)
    s = 88 + (idx % 3) * 4
    l = 34 + (idx % 4) * 2
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
    return f"#{r:02x}{g:02x}{b:02x}"

def _hex_to_rgba(hex_color: str, alpha: int = 60) -> list:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r, g, b, alpha]

def add_district_colors(geojson: dict, label_prefix: str, base_hue: str = "green") -> dict:
    """Add per-feature fill/line colors to GeoJSON properties for fast, vectorized rendering."""
    if not geojson or "features" not in geojson:
        return geojson

    # Collect district ids
    district_ids = []
    for feat in geojson["features"]:
        props = feat.get("properties", {}) or {}
        num = _district_number_from_props(props)
        if num:
            district_ids.append(num)

    unique_ids = sorted(set(district_ids), key=lambda x: (len(str(x)), str(x)))
    total = max(1, len(unique_ids))
    id_to_idx = {d: i for i, d in enumerate(unique_ids)}

    for feat in geojson["features"]:
        props = feat.setdefault("properties", {})
        num = _district_number_from_props(props)
        idx = id_to_idx.get(num, 0)
        hex_color = get_district_color(idx, total, base_hue=base_hue)

        # Keep the SAME yellow-ish hover via highlight_color (set on the layer), but give districts a green base.
        props["fill_color"] = _hex_to_rgba(hex_color, alpha=175 if "Assembly" in label_prefix else 165)
        props["line_color"] = _hex_to_rgba(hex_color, alpha=255)

        # Tooltip stays minimal
        props["tooltip_html"] = f"<b>{label_prefix}: {num}</b>" if num else f"<b>{label_prefix}</b>"

    return geojson




def get_impact_color(impact_value):
    """Get color based on impact level - Greenlining themed"""
    if impact_value >= 100:
        return [155, 28, 28, 200]  # Deep red
    elif impact_value >= 20:
        return [217, 119, 6, 200]  # Orange
    elif impact_value > 0:
        return [245, 158, 11, 200]  # Yellow
    else:
        return [0, 166, 81, 200]  # Greenlining green

def get_impact_radius(impact_value):
    """Calculate marker radius based on impact"""
    if impact_value == 0:
        return 800
    return min(3000, 500 + int((impact_value ** 0.7) * 100))

# Load data
df = load_data()
geojson_data = load_geojson_data()

# Sidebar controls (keeps the map as the first main-page element)
# ---- Filter state (no sidebar; widgets appear below the map) ----
def _init_filter_state():
    defaults = {
        'year': 'All Years',
        'impact': 'All Projects',
        'district_type': 'Assembly',
        'district': 'All Districts',
        'assembly': 'All Assembly Members',
        'senate': 'All Senators',
        'county': 'All Counties',
        'route': 'All Routes',
        'sort_by': 'Total Relocations',
        'enable_3d': True,
        'show_districts': True,
        'show_highways': False,
        'show_heatmap': True,
        'show_assembly': True,
        'show_senate': True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _flatten_unique_values(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Robust unique extraction for mixed dtypes (strings, NaN, lists/tuples/sets)."""
    out = []
    for v in frame[cols].to_numpy().ravel(order='K'):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, (list, tuple, set)):
            for item in v:
                if item is None or (isinstance(item, float) and pd.isna(item)):
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
        else:
            s = str(v).strip()
            if s:
                out.append(s)
    return sorted(set(out))

_init_filter_state()



@st.cache_data(ttl=1800, show_spinner=False)
def get_filter_options_cached(df: pd.DataFrame) -> dict:
    """Precompute dropdown options (cached) to reduce rerun work."""
    if df is None or df.empty:
        return {
            "years": ["All Years"],
            "counties": ["All Counties"],
            "routes": ["All Routes"],
            "assembly_members": ["All Assembly Members"],
            "senators": ["All Senators"],
        }

    opts = {}

    if "CCA_FY" in df.columns:
        yrs = df["CCA_FY"].dropna()
        try:
            yrs = yrs.astype(int)
        except Exception:
            yrs = yrs.astype(str)
        opts["years"] = ["All Years"] + sorted(pd.unique(yrs).tolist())
    else:
        opts["years"] = ["All Years"]

    opts["counties"] = ["All Counties"] + (sorted(pd.unique(df["County"].dropna().astype(str)).tolist()) if "County" in df.columns else [])
    opts["routes"] = ["All Routes"] + (sorted(pd.unique(df["Route"].dropna()).tolist()) if "Route" in df.columns else [])

    assembly_cols = [c for c in df.columns if c.startswith("Assemblymember")]
    if assembly_cols:
        am = pd.unique(pd.concat([df[c] for c in assembly_cols], ignore_index=True).dropna().astype(str)).tolist()
        opts["assembly_members"] = ["All Assembly Members"] + sorted(am)
    else:
        opts["assembly_members"] = ["All Assembly Members"]

    senate_cols = [c for c in df.columns if c.startswith("Senator")]
    if senate_cols:
        se = pd.unique(pd.concat([df[c] for c in senate_cols], ignore_index=True).dropna().astype(str)).tolist()
        opts["senators"] = ["All Senators"] + sorted(se)
    else:
        opts["senators"] = ["All Senators"]

    return opts

@st.cache_data(ttl=600, show_spinner=False)
def apply_filters_cached(
    df: pd.DataFrame,
    selected_year: str,
    impact_filter: str,
    district_type: str,
    selected_district,
    selected_assembly: str,
    selected_senate: str,
    selected_county: str,
    selected_route: str,
    sort_by: str,
) -> pd.DataFrame:
    """Fast, cached filtering + sorting. Keep this PURE (no Streamlit calls) for maximum cache hits."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df

    # Assembly/Senate filters (vectorized across columns)
    if selected_assembly != "All Assembly Members":
        assembly_cols = [c for c in out.columns if c.startswith("Assemblymember")]
        if assembly_cols:
            out = out[out[assembly_cols].eq(selected_assembly).any(axis=1)]

    if selected_senate != "All Senators":
        senate_cols = [c for c in out.columns if c.startswith("Senator")]
        if senate_cols:
            out = out[out[senate_cols].eq(selected_senate).any(axis=1)]

    # County / Route / Year
    if selected_county != "All Counties" and "County" in out.columns:
        out = out[out["County"] == selected_county]

    if selected_route != "All Routes" and "Route" in out.columns:
        # Support both numeric and string route values
        sroute = str(selected_route).strip()
        if sroute.isdigit():
            out = out[out["Route"] == int(sroute)]
        else:
            out = out[out["Route"].astype(str) == sroute]

    if selected_year != "All Years" and "CCA_FY" in out.columns:
        syear = str(selected_year).strip()
        if syear.isdigit():
            out = out[out["CCA_FY"] == int(syear)]
        else:
            out = out[out["CCA_FY"].astype(str) == syear]

    # Impact filter
    if impact_filter != "All Projects" and "Total_Relocations" in out.columns:
        tr = out["Total_Relocations"].fillna(0)
        if impact_filter == "High Impact (100+)":
            out = out[tr >= 100]
        elif impact_filter == "Medium Impact (20-100)":
            out = out[(tr >= 20) & (tr < 100)]
        elif impact_filter == "Low Impact (1-20)":
            out = out[(tr >= 1) & (tr < 20)]
        elif impact_filter == "No Impact":
            out = out[tr == 0]

    # Sorting
    if sort_by == "Total Relocations" and "Total_Relocations" in out.columns:
        out = out.sort_values("Total_Relocations", ascending=False)
    elif sort_by == "Homes Demolished" and "Num_Home_Demolished" in out.columns:
        out = out.sort_values("Num_Home_Demolished", ascending=False)
    elif sort_by == "Businesses Demolished" and "Num_Business_Demolished" in out.columns:
        out = out.sort_values("Num_Business_Demolished", ascending=False)
    elif sort_by == "Year" and "CCA_FY" in out.columns:
        out = out.sort_values("CCA_FY", ascending=False)
    elif sort_by == "County" and "County" in out.columns:
        out = out.sort_values("County")
    elif sort_by == "Route" and "Route" in out.columns:
        out = out.sort_values("Route")

    return out


@st.cache_data(ttl=600, show_spinner=False)
def prepare_map_df_cached(filtered_df: pd.DataFrame, enable_3d: bool) -> pd.DataFrame:
    """Prepare the dataframe for rendering (colors/radius/elevation/tooltips) using vectorized ops."""
    if filtered_df is None or filtered_df.empty:
        return pd.DataFrame()

    if "latitude" not in filtered_df.columns or "longitude" not in filtered_df.columns:
        return pd.DataFrame()

    map_df = filtered_df.loc[filtered_df["latitude"].notna(), :].copy()
    if map_df.empty:
        return map_df

    impact = map_df.get("Total_Relocations", pd.Series(0, index=map_df.index)).fillna(0).to_numpy()

    # Vectorized colors (RGBA)
    colors = np.empty((len(map_df), 4), dtype=int)
    colors[:] = [0, 166, 81, 200]  # default green
    colors[impact > 0] = [245, 158, 11, 200]     # yellow
    colors[impact >= 20] = [217, 119, 6, 200]    # orange
    colors[impact >= 100] = [155, 28, 28, 200]   # deep red
    map_df["color"] = colors.tolist()

    # Vectorized radius
    radius = np.where(
        impact == 0,
        800,
        np.minimum(3000, 500 + (np.power(impact, 0.7) * 100).astype(int)),
    )
    map_df["radius"] = radius.astype(int)

    # Vectorized elevation for 3D
    if enable_3d:
        elev = np.minimum(50000, 1000 + (impact * 300))
    else:
        elev = np.zeros_like(impact)
    map_df["elevation"] = elev.astype(int)

    # Vectorized tooltips (avoid .apply per-row)
    proj = map_df.get("Project", "N/A")
    proj = proj.fillna("N/A").astype(str)

    county = map_df.get("County", "N/A")
    county = county.fillna("N/A").astype(str)

    route = map_df.get("Route", "N/A")
    route = route.fillna("N/A").astype(str)

    homes = map_df.get("Num_Home_Demolished", 0).fillna(0).astype(int).astype(str)
    biz = map_df.get("Num_Business_Demolished", 0).fillna(0).astype(int).astype(str)
    total = map_df.get("Total_Relocations", 0).fillna(0).astype(int).astype(str)

    map_df["tooltip_html"] = (
        "<b>Project " + proj + "</b><br>"
        "County: " + county + "<br>"
        "Route: " + route + "<br>"
        "<hr style='margin: 8px 0;'>"
        "🏠 Homes: " + homes + "<br>"
        "🏢 Businesses: " + biz + "<br>"
        "📦 Total: <b>" + total + "</b> displaced"
    )

    return map_df

# Read current selections (widgets are defined later in the page)
selected_year = st.session_state.get('year', 'All Years')
impact_filter = st.session_state.get('impact', 'All Projects')
district_type = st.session_state.get('district_type', 'Assembly')
selected_district = st.session_state.get('district', 'All Districts')
selected_assembly = st.session_state.get('assembly', 'All Assembly Members')
selected_senate = st.session_state.get('senate', 'All Senators')
selected_county = st.session_state.get('county', 'All Counties')
selected_route = st.session_state.get('route', 'All Routes')
sort_by = st.session_state.get('sort_by', 'Total Relocations')
enable_3d = bool(st.session_state.get('enable_3d', True))
show_districts = bool(st.session_state.get('show_districts', True))
show_highways = bool(st.session_state.get('show_highways', False))
show_heatmap = bool(st.session_state.get('show_heatmap', True))
show_assembly = bool(st.session_state.get('show_assembly', True))
show_senate = bool(st.session_state.get('show_senate', True))

# Apply filters (cached for fast reruns)
if not df.empty:
    filtered_df = apply_filters_cached(
        df=df,
        selected_year=selected_year,
        impact_filter=impact_filter,
        district_type=district_type,
        selected_district=selected_district,
        selected_assembly=selected_assembly,
        selected_senate=selected_senate,
        selected_county=selected_county,
        selected_route=selected_route,
        sort_by=sort_by,
    )
    map_df = prepare_map_df_cached(filtered_df, enable_3d)
else:
    filtered_df = pd.DataFrame()
    map_df = pd.DataFrame()

if not map_df.empty:
        # NOTE: map_df is already fully prepared (color, radius, elevation, tooltip_html)
        # via prepare_map_df_cached(...). Avoid redoing per-row .apply() work here.

        # Calculate view state
        center_lat = float(map_df['latitude'].mean())
        center_lng = float(map_df['longitude'].mean())
        
        # Map section
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗺️ Interactive Project Map</div>', unsafe_allow_html=True)
        
        if enable_3d:
            st.markdown("""
            <div class="info-box">
                <strong>🎬 3D View Active!</strong> Displacement bars rise from the map - taller columns = 
                more displaced families. <em>Click and drag to rotate.</em> Hover over districts to see numbers.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>🗺️ 2D Map View:</strong> Hover over districts to see district numbers highlighted in yellow. 
                Hover over markers to see project details.
            </div>
            """, unsafe_allow_html=True)
        
        if show_highways:
            st.markdown("""
            <div class="warning-box">
                <strong>🛣️ Highway Network Overlay:</strong> Red lines show the California highway system. 
                These represent the infrastructure that displaced thousands of families and businesses.
            </div>
            """, unsafe_allow_html=True)
        
        # Build layers
        layers = []
        
        # Add district layers if enabled (2D only - no extrusion)
        if show_districts and show_assembly and 'assembly' in geojson_data:
            assembly_layer = pdk.Layer(
                'GeoJsonLayer',
                data=geojson_data['assembly'],
                opacity=0.3,
                stroked=True,
                filled=True,
                extruded=False,
                wireframe=True,
                get_fill_color='properties.fill_color',
                get_line_color='properties.line_color',
                line_width_min_pixels=2,
                pickable=True,
                auto_highlight=True,
                highlight_color=[255, 184, 28, 150]
            )
            layers.append(assembly_layer)
        
        if show_districts and show_senate and 'senate' in geojson_data:
            senate_layer = pdk.Layer(
                'GeoJsonLayer',
                data=geojson_data['senate'],
                opacity=0.25,
                stroked=True,
                filled=True,
                extruded=False,
                wireframe=True,
                get_fill_color='properties.fill_color',
                get_line_color='properties.line_color',
                line_width_min_pixels=2,
                line_width_max_pixels=3,
                pickable=True,
                auto_highlight=True,
                highlight_color=[255, 184, 28, 150]
            )
            layers.append(senate_layer)
        
        # Add highway network layer
        if show_highways and "highways" in geojson_data:
                highway_layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=geojson_data["highways"],
                    stroked=True,
                    filled=False,
                    get_line_color=[255, 0, 0, 220],   # RED
                    get_line_width=200,               # thickness (units depend on width_scale)
                    line_width_scale=1,
                    line_width_min_pixels=4,          # thick on screen
                    line_width_max_pixels=20,
                    pickable=True,
                    auto_highlight=True,
                    highlight_color=[255, 255, 0, 220],
                )
            
                layers.append(highway_layer)
        
        # Add heatmap if enabled
        if show_heatmap and not map_df.empty:
            heatmap_layer = pdk.Layer(
                'HeatmapLayer',
                data=map_df,
                get_position=['longitude', 'latitude'],
                get_weight='Total_Relocations',
                radiusPixels=60,
                intensity=1,
                threshold=0.05,
                color_range=[
                    [0, 166, 81, 25],
                    [76, 187, 135, 85],
                    [255, 184, 28, 127],
                    [217, 119, 6, 170],
                    [155, 28, 28, 255]
                ]
            )
            layers.append(heatmap_layer)
        
        # Add project markers - use ColumnLayer for 3D or ScatterplotLayer for 2D
        if not map_df.empty:
            if enable_3d and show_projects:
                # 2D footprint layer (subtle) so impacts remain visible when zoomed out
                footprint_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position=['longitude', 'latitude'],
                    get_color='color',
                    get_radius='radius',
                    opacity=0.22,
                    stroked=False,
                    filled=True,
                    radius_scale=1.8,
                    radius_min_pixels=2,
                    radius_max_pixels=60,
                    pickable=False,
                )
                layers.append(footprint_layer)

                # 3D Column layer for dramatic effect
                column_layer = pdk.Layer(
                    'ColumnLayer',
                    data=map_df,
                    get_position=['longitude', 'latitude'],
                    get_fill_color='color',
                    get_elevation='elevation',
                    elevation_scale=1.6,  # slightly boosted to stay legible at lower zooms
                    radius=900,
                    pickable=True,
                    auto_highlight=True,
                    extruded=True,
                    coverage=1,
                    get_line_color=[255, 255, 255, 110],
                )
                layers.append(column_layer)
            else:
                # 2D Scatterplot layer
                scatterplot_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position=['longitude', 'latitude'],
                    get_color='color',
                    get_radius='radius',
                    pickable=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    radius_scale=1,
                    radius_min_pixels=5,
                    radius_max_pixels=50,
                    line_width_min_pixels=1,
                    get_line_color=[255, 255, 255, 180]
                )
                layers.append(scatterplot_layer)
        
        # Create view that fits *all* visible projects by default (better "at-a-glance" coverage statewide)
        try:
            view_state = pdk.data_utils.compute_view(map_df[["longitude", "latitude"]])
            # Add a bit more padding (zoom out slightly) for statewide context
            view_state.zoom = max(4.5, float(view_state.zoom) - (0.7 if enable_3d else 0.5))
            view_state.pitch = 45 if enable_3d else 0
            view_state.bearing = 0
        except Exception:
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lng,
                zoom=6,
                pitch=45 if enable_3d else 0,
                bearing=0,
            )        # Use Carto basemap (positron for light)
        map_style = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
        
        deck = pdk.Deck(
            map_style=map_style,
            initial_view_state=view_state,
            layers=layers,
            tooltip={
                "html": "{tooltip_html}",
                "style": {
                    "backgroundColor": "white",
                    "color": "#1C1C1C",
                    "fontSize": "14px",
                    "padding": "8px 12px",
                    "borderRadius": "6px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
                    "fontFamily": "Inter, sans-serif",
                },
            },
            parameters={
                "clearColor": [0.95, 0.97, 0.95, 1] if not enable_3d else [0.1, 0.1, 0.1, 1]
            }
        )
        
        # In-map legend overlay (positioned over the map)
        
        st.markdown(
        
            '''
        
            <style>
        
              .map-wrap { position: relative; }
        
              .map-wrap .map-legend {
        
                position: absolute;
        
                top: 16px;
        
                right: 16px;
        
                z-index: 999;
        
                width: 250px;
        
                background: rgba(255, 255, 255, 0.92);
        
                border: 1px solid rgba(0, 166, 81, 0.25);
        
                border-radius: 14px;
        
                padding: 12px 12px 10px 12px;
        
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        
                color: #1C1C1C;
        
                backdrop-filter: blur(6px);
        
              }
        
              .map-wrap .map-legend-title { font-weight: 800; font-size: 14px; margin-bottom: 8px; color: #007A33; }
        
              .map-wrap .map-legend-row { display: flex; align-items: center; gap: 10px; font-size: 13px; margin: 6px 0; line-height: 1.2; }
        
              .map-wrap .swatch { width: 14px; height: 14px; border-radius: 4px; display: inline-block; border: 1px solid rgba(0,0,0,0.15); }
        
              .map-wrap .swatch-red { background: rgba(155, 28, 28, 0.95); }
        
              .map-wrap .swatch-orange { background: rgba(217, 119, 6, 0.95); }
        
              .map-wrap .swatch-yellow { background: rgba(245, 158, 11, 0.95); }
        
              .map-wrap .swatch-green { background: rgba(0, 166, 81, 0.95); }
        
              .map-wrap .swatch-blue { background: rgba(70, 140, 220, 0.55); }
        
              .map-wrap .swatch-lightgreen { background: rgba(110, 185, 120, 0.55); }
        
              .map-wrap .swatch-hwy { background: rgba(255, 0, 0, 0.75); }
        
              .map-wrap .map-legend-divider { height: 1px; background: rgba(0,0,0,0.08); margin: 10px 0 8px 0; }
        
              .map-wrap .map-legend-tip { margin-top: 8px; font-size: 12px; color: #54565B; }
        
            </style>
        
            ''',
        
            unsafe_allow_html=True,
        
        )

        
        legend_title = "Map Key (Height = Impact in 3D)" if enable_3d else "Map Key"

        
        st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
        
        st.markdown(
        
            f'''
        
            <div class="map-legend" aria-label="Map legend">
        
              <div class="map-legend-title">🗝️ {legend_title}</div>
        
              <div class="map-legend-row"><span class="swatch swatch-red"></span> High impact (100+ displaced)</div>
        
              <div class="map-legend-row"><span class="swatch swatch-orange"></span> Medium impact (20–99)</div>
        
              <div class="map-legend-row"><span class="swatch swatch-yellow"></span> Low impact (1–19)</div>
        
              <div class="map-legend-row"><span class="swatch swatch-green"></span> No impact (0)</div>
        
              <div class="map-legend-divider"></div>
        
              <div class="map-legend-row"><span class="swatch swatch-blue"></span> Assembly districts</div>
        
              <div class="map-legend-row"><span class="swatch swatch-lightgreen"></span> Senate districts</div>
        
              <div class="map-legend-row"><span class="swatch swatch-hwy"></span> Highways</div>
        
              <div class="map-legend-tip">Tip: Hover for details • Drag to pan • Scroll to zoom</div>
        
            </div>
        
            ''',
        
            unsafe_allow_html=True,
        
        )

        
        st.pydeck_chart(deck, width="stretch")
        
        st.markdown('</div>', unsafe_allow_html=True)
 # Legend section
        st.markdown(f"#### 📍 Map Legend {'(Height = Impact in 3D)' if enable_3d else ''}")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFB81C 0%, #FF9800 100%); 
                    padding: 10px 15px; border-radius: 8px; color: #1C1C1C; text-align: center; 
                    margin-bottom: 15px; font-weight: 600; border: 2px solid #FF9800;">
            ✨ <strong>Hover over districts</strong> to see district numbers | <strong>Hover over markers</strong> for project details
        </div>
        """, unsafe_allow_html=True)
   
        
st.markdown("## ⚙️ Filters & Layers")
with st.container():
    if df.empty:
        st.info('No project data loaded yet.')
    else:

        # Improved controls: map layer toggles first (more accessible) + cached option lists for faster reruns
        control_tabs = st.tabs(["🗺️ Layers", "🔎 Filters"])

        with control_tabs[0]:
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                st.checkbox("🗺️ Show District Layers", key="show_districts")
            with r1c2:
                st.checkbox("🛣️ Show Highways", key="show_highways")
            with r1c3:
                st.checkbox("🔥 Show Heatmap", key="show_heatmap")

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                st.checkbox("🏢 Enable 3D Bars", key="enable_3d")
            with r2c2:
                st.checkbox("🏛️ Highlight Assembly", key="show_assembly")
            with r2c3:
                st.checkbox("🏛️ Highlight Senate", key="show_senate")

            st.caption("Tip: toggles update the map immediately.")

        with control_tabs[1]:
            opts = get_filter_options_cached(df)

            years = opts.get("years", ["All Years"])
            counties = opts.get("counties", ["All Counties"])
            routes = opts.get("routes", ["All Routes"])
            assembly_members = opts.get("assembly_members", ["All Assembly Members"])
            senators = opts.get("senators", ["All Senators"])

            impact_options = ["All Projects", "High Impact (100+)", "Medium Impact (20-100)", "Low Impact (1-20)", "No Impact"]
            district_type_options = ["Assembly", "Senate"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.selectbox("📅 Year", years, key="year")
            with c2:
                st.selectbox("🎯 Impact Level", impact_options, key="impact")
            with c3:
                st.selectbox("🗳️ District Type", district_type_options, key="district_type")

            # District dropdown depends on type
            if st.session_state.get("district_type", "Assembly") == "Assembly":
                district_col = "AssemblyDistrict" if "AssemblyDistrict" in df.columns else None
                label = "🏛️ Assembly District"
            else:
                district_col = "SenateDistrict" if "SenateDistrict" in df.columns else None
                label = "🏛️ Senate District"

            if district_col:
                districts = ["All Districts"] + sorted(pd.unique(df[district_col].dropna()).tolist())
                st.selectbox(label, districts, key="district")
            else:
                st.selectbox(label, ["All Districts"], key="district", disabled=True)

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                st.selectbox("🏛️ Assembly Member", assembly_members, key="assembly")
            with f2:
                st.selectbox("🏛️ Senator", senators, key="senate")
            with f3:
                st.selectbox("📍 County", counties, key="county")
            with f4:
                st.selectbox("🛣️ Route", routes, key="route")

            sort_options = ["Total Relocations", "Homes Demolished", "Businesses Demolished", "Year", "County", "Route"]
            st.selectbox("↕️ Sort projects by", sort_options, key="sort_by")

        # Data table
        st.markdown('<div class="section-title">📋 Project Details</div>', unsafe_allow_html=True)

        show_table = st.checkbox('📋 Show project table', value=True, key='show_table')
        if show_table:
            display_columns = [
                'Project', 'County', 'Assemblymember 1', 'Senator 1', 'Route', 'CCA_FY',
                'Num_Home_Demolished', 'Num_Business_Demolished', 'Total_Relocations'
            ]
            available_columns = [col for col in display_columns if col in filtered_df.columns]

            if available_columns and not filtered_df.empty:
                display_df = filtered_df[available_columns].copy()

                styled_df = display_df.style.format({
                    'Num_Home_Demolished': '{:.0f}',
                    'Num_Business_Demolished': '{:.0f}',
                    'Total_Relocations': '{:.0f}'
                }).background_gradient(
                    subset=['Total_Relocations'],
                    cmap='YlOrRd',
                    vmin=0,
                    vmax=filtered_df['Total_Relocations'].max() if filtered_df['Total_Relocations'].max() > 0 else 1
                )

                st.dataframe(styled_df, use_container_width=True, height=400)
            else:
                st.info("No rows to display for the current filters.")
        else:
            st.caption('Table is off by default for faster loads. Turn it on if you need to scan rows.')

        # Footer
        st.markdown("""
        <div class="info-box">
            <strong>About This Data:</strong> This visualization tracks highway construction projects 
            across California and their impact on communities, including homes demolished, businesses 
            displaced, and total relocations. Data represents documented impacts from highway 
            expansion projects.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #54565B;">
            <p><strong>Powered by The Greenlining Institute</strong></p>
            <p style="font-size: 0.9rem;">Fighting for racial and economic justice since 1993</p>
        </div>
        """, unsafe_allow_html=True)
    
    if map_df.empty:
        st.warning("⚠️ No projects with valid coordinates match your current filters. Try adjusting your selection.")



