import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import plotly.express as px
import requests
import os
from datetime import datetime, timedelta

# 1. CONFIGURACIÓN (Debe ser la primera línea)
st.set_page_config(page_title="Pacomarca Scientific", layout="wide", page_icon="🧬")

# 2. AUTENTICACIÓN SEGURA (Lectura del Token)
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
    st.error("⚠️ Error de conexión GEE. Revisa los Secrets.")
    st.stop()

# 3. ESTILOS CSS (Mínimos y Seguros para evitar pantalla blanca)
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .big-font { font-size: 30px !important; font-weight: bold; color: #1f2937; }
    .label-font { font-size: 14px; color: #6b7280; text-transform: uppercase;}
</style>
""", unsafe_allow_html=True)

# 4. FUNCIONES
@st.cache_data
def get_chart_data(lat, lon, years):
    roi = ee.Geometry.Point([lon, lat]).buffer(2000)
    end = datetime.now()
    start = end - timedelta(days=365*years)
    # Usamos MODIS para la serie de tiempo (rápido y ligero)
    ds = ee.ImageCollection('MODIS/006/MOD13Q1').filterDate(start, end).filterBounds(roi).select('NDVI')
    data = ds.map(lambda img: ee.Feature(None, {'d': img.date().format('YYYY-MM-dd'), 'v': img.reduceRegion(ee.Reducer.mean(), roi, 250).get('NDVI')})).getInfo()
    
    df = pd.DataFrame([f['properties'] for f in data['features']])
    if not df.empty:
        df['d'] = pd.to_datetime(df['d'])
        df['v'] = df['v']/10000
        df['biomasa'] = df['v'] * 2800
        return df.sort_values('d')
    return pd.DataFrame()

# 5. INTERFAZ DE USUARIO
st.title("🧬 PACOMARCA: Monitor Científico")

# --- Barra Lateral ---
with st.sidebar:
    st.header("Parámetros")
    c_lat = st.number_input("Latitud", value=-14.85000, format="%.5f")
    c_lon = st.number_input("Longitud", value=-70.92000, format="%.5f")
    
    # Selector de capas (Requisito VicuñaPastos)
    layer = st.selectbox(
        "Capa de Análisis", 
        ["Biomasa (Kg/Ha)", "Clasificación (IA)", "Radar S1 (Nubes)", "NDVI"]
    )
    years = st.slider("Historial (Años)", 5, 23, 20)
    st.info("Sistema conectado a Google Earth Engine")

# --- Cuerpo Principal ---
col_map, col_stats = st.columns([3, 1])

with col_map:
    # Configuración del Mapa
    m = geemap.Map(center=[c_lat, c_lon], zoom=14)
    roi = ee.Geometry.Point([c_lon, c_lat]).buffer(2000)
    
    # LOGICA DE CAPAS (Aquí es donde estaba incompleto el código anterior)
    
    # 1. Biomasa (Tu requisito principal)
    if "Biomasa" in layer:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2:
            viz = s2.normalizedDifference(['B8', 'B4']).multiply(2800)
            m.addLayer(viz, {'min':0, 'max':2500, 'palette':['#ffffe5', '#f7fcb9', '#addd8e', '#41ab5d', '#005a32']}, "Biomasa Est.")
    
    # 2. Radar S1 (Requisito 'CambioAlto' del documento - ve a través de nubes)
    elif "Radar" in layer:
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(datetime.now()-timedelta(days=30), datetime.now()).first()
        if s1:
            m.addLayer(s1.select('VV'), {'min':-25, 'max':5}, "Radar Sentinel-1")
    
    # 3. Clasificación (Requisito 'VicuñaPastos' - segmentación simple)
    elif "Clasificación" in layer:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2:
            # Clustering simple K-Means para simular clasificación de hábitat
            training = s2.sample(region=roi, scale=10, numPixels=1000)
            clusterer = ee.Clusterer.wekaKMeans(3).train(training)
            result = s2.cluster(clusterer)
            m.addLayer(result, {'min':0, 'max':2, 'palette':['red', 'yellow', 'green']}, "Hábitat IA")
            
    # 4. NDVI
    else:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2:
            m.addLayer(s2.normalizedDifference(['B8', 'B4']), {'min':0, 'max':0.8, 'palette':['red', 'yellow', 'green']}, "NDVI")

    m.addLayer(roi, {'color':'white', 'width': 2}, "Zona de Estudio")
    m.to_streamlit(height=500)

with col_stats:
    # Procesamiento de datos históricos
    df = get_chart_data(c_lat, c_lon, years)
    
    if not df.empty:
        val_actual = df['biomasa'].iloc[-1]
        promedio = df['biomasa'].mean()
        
        # Tarjeta de KPI limpia
        st.markdown(f"""
        <div class="metric-box">
            <div class="label-font">Promedio Actual</div>
            <div class="big-font">{val_actual:.0f}</div>
            <div style="color:#2563EB; font-weight:bold;">Kg/Ha</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Espacio
        
        # Sistema de Alertas (Requisito 'CambioAlto')
        if val_actual < 500:
            st.error("🚨 **CRÍTICO**\n\nDegradación severa detectada.")
        elif val_actual < 1500:
            st.warning("⚠️ **ALERTA**\n\nNiveles bajo el promedio.")
        else:
            st.success("✅ **ÓPTIMO**\n\nCondición saludable.")
            
        st.caption(f"Promedio histórico: {promedio:.0f} Kg/Ha")

# Gráfico de Tendencia
if not df.empty:
    st.subheader("Dinámica Temporal (20 Años)")
    fig = px.area(df, x='d', y='biomasa')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20,r=20,t=10,b=20),
        yaxis_title="Biomasa (Kg/Ha)",
        xaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:grey;'>Desarrollado por Jhon Monroy | Experto en Informática</div>", unsafe_allow_html=True)
