import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import plotly.express as px
import requests
import os
from datetime import datetime, timedelta

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Pacomarca Scientific", layout="wide", page_icon="🧬")

# 2. CSS "NUCLEAR" (FORZADO DE MODO CLARO ABSOLUTO)
st.markdown("""
<style>
    /* Forzar variables de color globales */
    :root {
        --primary-color: #2563EB;
        --background-color: #FFFFFF;
        --secondary-background-color: #F8FAFC;
        --text-color: #000000;
        --font: "sans-serif";
    }
    
    /* Fondo general y Texto */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Sidebar Blanco */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Todos los textos a Negro */
    p, h1, h2, h3, label, li, span, div {
        color: #000000 !important;
    }
    
    /* Tarjetas de Métricas */
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Ajuste de Inputs para que se vean bien en fondo blanco */
    .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. AUTENTICACIÓN SEGURA
try:
    if "earth_engine" in st.secrets:
        credentials = st.secrets["earth_engine"]["token"]
        home = os.path.expanduser("~")
        path = os.path.join(home, ".config", "earthengine")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "credentials"), "w") as f:
            f.write(credentials)
    ee.Initialize(project='egresados-q9tr')
except Exception as e:
    st.error(f"⚠️ Error GEE: {e}")
    st.stop()

# 4. FUNCIONES
@st.cache_data
def get_chart_data(lat, lon, years):
    roi = ee.Geometry.Point([lon, lat]).buffer(2000)
    end = datetime.now()
    start = end - timedelta(days=365*years)
    ds = ee.ImageCollection('MODIS/006/MOD13Q1').filterDate(start, end).filterBounds(roi).select('NDVI')
    data = ds.map(lambda img: ee.Feature(None, {'d': img.date().format('YYYY-MM-dd'), 'v': img.reduceRegion(ee.Reducer.mean(), roi, 250).get('NDVI')})).getInfo()
    df = pd.DataFrame([f['properties'] for f in data['features']])
    if not df.empty:
        df['d'] = pd.to_datetime(df['d'])
        df['v'] = df['v']/10000
        df['biomasa'] = df['v'] * 2800
        return df.sort_values('d')
    return pd.DataFrame()

# 5. INTERFAZ
st.title("🧬 PACOMARCA: Monitor Científico")

# Sidebar
with st.sidebar:
    st.header("Parámetros")
    c_lat = st.number_input("Latitud", value=-14.85000, format="%.5f")
    c_lon = st.number_input("Longitud", value=-70.92000, format="%.5f")
    
    # Capas (Simplificadas para estabilidad)
    layer = st.selectbox(
        "Capa de Análisis", 
        ["Biomasa (Kg/Ha)", "Clasificación (Hábitat)", "Radar S1 (Nubes)", "NDVI"]
    )
    years = st.slider("Historial (Años)", 5, 23, 20)

# Columnas Principales
col_map, col_stats = st.columns([3, 1])

with col_map:
    m = geemap.Map(center=[c_lat, c_lon], zoom=14)
    roi = ee.Geometry.Point([c_lon, c_lat]).buffer(2000)
    
    # --- LÓGICA DE CAPAS ROBUSTA (SIN ERRORES) ---
    
    # 1. Biomasa (Tu requisito principal)
    if "Biomasa" in layer:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2:
            viz = s2.normalizedDifference(['B8', 'B4']).multiply(2800)
            m.addLayer(viz, {'min':0, 'max':2500, 'palette':['#ffffe5', '#f7fcb9', '#addd8e', '#41ab5d', '#005a32']}, "Biomasa")
    
    # 2. Radar S1 (Requisito CambioAlto)
    elif "Radar" in layer:
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(datetime.now()-timedelta(days=30), datetime.now()).first()
        if s1:
            m.addLayer(s1.select('VV'), {'min':-25, 'max':5}, "Radar Sentinel-1")
    
    # 3. Clasificación (Requisito VicuñaPastos - VERSIÓN ESTABLE)
    # Reemplazamos K-Means (que fallaba) por Clasificación por Umbrales (No falla)
    elif "Clasificación" in layer:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2:
            ndvi = s2.normalizedDifference(['B8', 'B4'])
            # Definimos clases: 0=Suelo, 1=Pasto Pobre, 2=Pasto Rico
            classified = ee.Image(0).where(ndvi.gt(0.2), 1).where(ndvi.gt(0.5), 2).clip(roi)
            m.addLayer(classified, {'min':0, 'max':2, 'palette':['red', 'yellow', 'green']}, "Hábitat Clasificado")
            
    # 4. NDVI
    else:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2:
            m.addLayer(s2.normalizedDifference(['B8', 'B4']), {'min':0, 'max':0.8, 'palette':['red', 'yellow', 'green']}, "NDVI")

    m.addLayer(roi, {'color':'black', 'width': 2}, "Zona de Estudio")
    m.to_streamlit(height=500)

with col_stats:
    df = get_chart_data(c_lat, c_lon, years)
    
    if not df.empty:
        val_actual = df['biomasa'].iloc[-1]
        promedio = df['biomasa'].mean()
        
        # Tarjeta KPI Blanca y Negra
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:14px; font-weight:bold; color:#4B5563;">PROMEDIO ACTUAL</div>
            <div style="font-size:40px; font-weight:900; color:#000000;">{val_actual:.0f}</div>
            <div style="color:#2563EB; font-weight:bold;">Kg/Ha</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Alertas
        if val_actual < 500:
            st.error("🚨 CRÍTICO: Degradación severa.")
        elif val_actual < 1500:
            st.warning("⚠️ ALERTA: Bajo el promedio.")
        else:
            st.success("✅ ÓPTIMO: Condición saludable.")
            
        st.caption(f"Promedio histórico (20 años): {promedio:.0f} Kg/Ha")

# Gráfico Forzado a Blanco
if not df.empty:
    st.subheader("Dinámica Temporal")
    fig = px.area(df, x='d', y='biomasa')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20,r=20,t=10,b=20),
        xaxis=dict(showgrid=False, title="", tickfont=dict(color='black')),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="Kg/Ha", tickfont=dict(color='black')),
        font=dict(color='black') # Fuerza texto negro en gráfico
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:black;'>Desarrollado por Jhon Monroy | Experto en Informática</div>", unsafe_allow_html=True)
