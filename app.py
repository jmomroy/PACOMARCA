import streamlit as st
import ee  # NOTA: Se instala como 'earthengine-api', pero se importa como 'ee'
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression # NOTA: Se instala como 'scikit-learn'
from sklearn.metrics import r2_score

# ==========================================
# 1. CONFIGURACIÓN "PACOMARCA SUITE - HIGH CONTRAST"
# ==========================================
st.set_page_config(
    page_title="Pacomarca Scientific Suite", 
    layout="wide", 
    page_icon="🧬", 
    initial_sidebar_state="expanded"
)

# --- CSS MAESTRO (FORZADO DE TEXTO NEGRO) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* 1. REGLA DE ORO: TODO EL TEXTO DEBE SER OSCURO */
    html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, span, label, div, li, a, button {
        font-family: 'Inter', sans-serif;
        color: #0F172A !important; /* AZUL NOCHE PROFUNDO */
    }

    /* 2. FONDO BLANCO PURO */
    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* 3. SIDEBAR (PANEL LATERAL) */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #CBD5E1;
    }
    
    /* Headers del Sidebar */
    .sidebar-label {
        color: #334155 !important;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    /* 4. INPUTS Y SELECTORES (Corrección visual) */
    /* El fondo del input blanco y el texto negro */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input {
        background-color: #FFFFFF !important;
        border-color: #94A3B8 !important;
        color: #000000 !important;
        font-weight: 600;
    }
    /* El texto dentro de los selectores */
    div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* 5. TARJETAS (CARDS) */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* 6. KPIs Y MÉTRICAS */
    .kpi-value {
        font-size: 36px;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -1px;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748B !important;
        text-transform: uppercase;
    }

    /* 7. PESTAÑAS (TABS) DE ALTO CONTRASTE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        font-size: 15px;
        font-weight: 600;
        color: #64748B !important; /* Gris oscuro inactivo */
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important; /* Azul activo */
        border-bottom: 3px solid #2563EB !important;
        background-color: #FFFFFF !important;
    }

    /* 8. FOOTER */
    .footer {
        text-align: center;
        padding: 30px;
        color: #64748B !important;
        font-size: 13px;
        font-weight: 600;
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
    }
    
    /* Corrección Radio Buttons */
    div[role="radiogroup"] label p {
        font-weight: 600 !important;
    }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONEXIÓN GEE
# ==========================================
PROJECT_ID = 'egresados-q9tr'
try:
    ee.Initialize(project=PROJECT_ID)
except:
    try:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    except:
        st.error("⚠️ Error de Conexión a Motores Satelitales")
        st.stop()

# ==========================================
# 3. FUNCIONES DE PROCESAMIENTO
# ==========================================
@st.cache_data
def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        return requests.get(url).json()
    except: return None

@st.cache_data
def get_chart_data(lat, lon, years):
    roi_internal = ee.Geometry.Point([lon, lat]).buffer(2000)
    end = datetime.now()
    start = end - timedelta(days=365*years)
    
    # MODIS NDVI Diaria
    ds = ee.ImageCollection('MODIS/006/MOD13Q1') \
           .filterDate(start, end) \
           .filterBounds(roi_internal) \
           .select('NDVI')
           
    data = ds.map(lambda img: ee.Feature(None, {
        'd': img.date().format('YYYY-MM-dd'), 
        'v': img.reduceRegion(ee.Reducer.mean(), roi_internal, 250).get('NDVI')
    })).getInfo()
    
    df = pd.DataFrame([f['properties'] for f in data['features']])
    if not df.empty:
        df['d'] = pd.to_datetime(df['d'])
        df['v'] = df['v']/10000
        df = df.sort_values('d')
        # Cálculos derivados
        df['biomasa'] = df['v'] * 2800 # Modelo empírico (Kg/Ha)
        return df
    return pd.DataFrame()

# ==========================================
# 4. BARRA LATERAL (CONTROLES)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063835.png", width=50)
    st.markdown("### PACOMARCA\n**SCIENTIFIC SUITE**")
    
    st.markdown('<div class="sidebar-label">📍 COORDENADAS</div>', unsafe_allow_html=True)
    c_lat = st.number_input("Latitud", value=-14.85000, format="%.5f")
    c_lon = st.number_input("Longitud", value=-70.92000, format="%.5f")
    roi = ee.Geometry.Point([c_lon, c_lat]).buffer(2000)

    st.markdown('<div class="sidebar-label">🛰️ SENSORES Y MODELOS</div>', unsafe_allow_html=True)
    
    # NUEVOS ÍNDICES SEGÚN DOCUMENTO VICUÑAPASTOS
    layer_mode = st.radio(
        "Variable de Estudio:",
        ["Biomasa (Kg/Ha)", "Clasificación (IA)", "Radar S1 (Estructura)", "NDVI (Vigor)", "EVI (Alta Densidad)"],
        captions=[
            "Estimación de Forraje", 
            "Segmentación de Hábitat", 
            "Penetración de Nubes",
            "Vigor Vegetal Estándar",
            "Índice Mejorado"
        ],
        index=0
    )
    
    st.markdown('<div class="sidebar-label">⏳ VENTANA TEMPORAL</div>', unsafe_allow_html=True)
    years = st.slider("Años de Análisis", 5, 23, 20)

# ==========================================
# 5. DASHBOARD PRINCIPAL
# ==========================================

# --- HEADER DE CLIMA (Estilo Tarjeta Blanca) ---
w = get_weather(c_lat, c_lon)
if w and 'current' in w:
    temp = w['current']['temperature_2m']
    hum = w['current']['relative_humidity_2m']
    rain = w['current']['rain']
else:
    temp, hum, rain = "--", "--", "--"

st.markdown(f"""
<div class="card" style="padding: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h1 style="margin:0; font-size: 28px; font-weight: 800; color: #0F172A;">Fundo Pacomarca</h1>
        <p style="margin:0; font-size: 15px; color: #475569; font-weight: 500;">Centro de Investigación y Producción de Camélidos</p>
    </div>
    <div style="display: flex; gap: 40px; text-align: right;">
        <div>
            <div class="kpi-value" style="font-size: 26px;">{temp}°</div>
            <div class="kpi-label">TEMP. AIRE</div>
        </div>
        <div>
            <div class="kpi-value" style="font-size: 26px; color: #2563EB;">{hum}%</div>
            <div class="kpi-label">H. RELATIVA</div>
        </div>
        <div>
            <div class="kpi-value" style="font-size: 26px; color: #059669;">{rain}</div>
            <div class="kpi-label">LLUVIA (MM)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- GRID VISUAL (MAPA + KPI CIENTÍFICO) ---
c1, c2 = st.columns([3, 1])

with c1:
    # Contenedor del Mapa
    with st.container():
        st.markdown('<div class="card" style="padding: 0; overflow: hidden; border: 1px solid #E2E8F0;">', unsafe_allow_html=True)
        m = geemap.Map(center=[c_lat, c_lon], zoom=14, basemap="HYBRID")
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(datetime.now()-timedelta(days=60), datetime.now()).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        val_disp = 0
        unit = ""
        legend = ""
        
        if s2:
            if layer_mode == "Biomasa (Kg/Ha)":
                ndvi = s2.normalizedDifference(['B8', 'B4'])
                img = ndvi.multiply(2800).rename('Biomasa') # Factor Pacomarca
                vis = {'min': 0, 'max': 2500, 'palette': ['#ffffe5', '#f7fcb9', '#addd8e', '#41ab5d', '#005a32']}
                legend = "Disponibilidad de Forraje"
                unit = "Kg/Ha"
            elif layer_mode == "Clasificación (IA)":
                input_img = s2.select(['B4', 'B3', 'B2', 'B8'])
                training = input_img.sample(region=roi, scale=10, numPixels=1000)
                clusterer = ee.Clusterer.wekaKMeans(3).train(training)
                img = input_img.cluster(clusterer)
                vis = {'min': 0, 'max': 2, 'palette': ['#d73027', '#fee08b', '#1a9850']} # Rojo=Suelo, Verde=Pasto
                legend = "Zonificación de Hábitat"
                unit = "Clase"
            elif layer_mode == "Radar S1 (Estructura)":
                s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(datetime.now()-timedelta(days=30), datetime.now()).first()
                if s1:
                    img = s1.select('VV')
                    vis = {'min': -25, 'max': 5}
                    legend = "Retrodispersión (SAR)"
                    unit = "dB"
                else:
                    img = ee.Image(0)
                    vis = {}
            elif layer_mode == "EVI":
                img = s2.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {'NIR': s2.select('B8'), 'RED': s2.select('B4'), 'BLUE': s2.select('B2')})
                vis = {'min': 0, 'max': 1, 'palette': ['white', 'green']}
                legend = "Índice EVI"
                unit = "idx"
            else: # NDVI
                img = s2.normalizedDifference(['B8', 'B4'])
                vis = {'min': 0, 'max': 0.8, 'palette': ['#d73027', '#fdae61', '#d9ef8b', '#1a9850']}
                legend = "Vigor Vegetal"
                unit = "idx"

            m.addLayer(img.clip(roi), vis, layer_mode)
            m.addLayer(roi, {'color': 'white', 'width': 2, 'fillColor': '00000000'}, "Límite")
            try: val_disp = img.reduceRegion(ee.Reducer.mean(), roi, 20).getInfo().get(list(img.bandNames().getInfo())[0], 0)
            except: val_disp = 0

        m.to_streamlit(height=480)
        st.markdown('</div>', unsafe_allow_html=True)

with c2:
    # Tarjeta KPI Científica
    st.markdown(f"""
    <div class="card" style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
        <div class="kpi-label" style="margin-bottom: 10px;">PROMEDIO ZONAL</div>
        <div class="kpi-value" style="font-size: 52px; color: #0F172A;">{val_disp:.2f}</div>
        <div style="font-size: 16px; font-weight: 700; color: #2563EB; margin-bottom: 20px;">{unit}</div>
        <div style="height: 6px; width: 100%; background: #F1F5F9; border-radius: 3px;">
            <div style="height: 100%; width: 60%; background: #0F172A; border-radius: 3px;"></div>
        </div>
        <p style="margin-top: 20px; font-size: 13px; font-weight: 600; color: #64748B;">
            {legend} <br> Dato satelital en tiempo real.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Alerta de Cambio (Requisito CambioAlto)
    if layer_mode == "Biomasa (Kg/Ha)":
        if val_disp < 500:
            st.error("🚨 ALERTA CRÍTICA: Baja disponibilidad de forraje.")
        elif val_disp < 1500:
            st.warning("⚠️ ALERTA TEMPRANA: Estrés detectado.")
        else:
            st.success("✅ ESTABLE: Capacidad de carga adecuada.")

# ANALYTICS
st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📊 Análisis de Tendencias y Alertas")
st.markdown("<p style='color:#64748B; font-weight:500;'>Evaluación de dinámica temporal y detección de anomalías.</p>", unsafe_allow_html=True)

df = get_chart_data(c_lat, c_lon, years)

if not df.empty:
    tab1, tab2 = st.tabs(["Dinámica Temporal", "Monitor de Alertas"])
    
    # TAB 1: GRÁFICO HISTÓRICO
    with tab1:
        variable_plot = 'biomasa' if "Biomasa" in layer_mode else 'v'
        y_label = "Biomasa (Kg/Ha)" if "Biomasa" in layer_mode else "Valor Índice"
        
        # CONFIGURACIÓN GRÁFICA NEGRA PARA EJES Y TEXTO
        fig = px.area(df, x='d', y=variable_plot, height=350)
        fig.update_traces(line_color='#2563EB', fillcolor='rgba(37, 99, 235, 0.1)')
        fig.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(
                showgrid=False, 
                tickfont=dict(color='black', size=12), # Eje X Negro
                title=""
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#f1f5f9', 
                tickfont=dict(color='black', size=12), # Eje Y Negro
                title=dict(text=y_label, font=dict(color='black', size=14)) # Título Eje Negro
            ),
            font=dict(family="Inter", color="black") # Fuente General Negra
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: DETECCIÓN DE ALERTAS (NUEVO SERVICIO)
    with tab2:
        # Cálculo de Anomalía
        current_val = df[variable_plot].iloc[-1]
        hist_mean = df[variable_plot].mean()
        anomaly = ((current_val - hist_mean) / hist_mean) * 100
        
        col_alert, col_metric = st.columns([2, 1])
        
        with col_alert:
            if anomaly < -15:
                st.error(f"🚨 **ALERTA DETECTADA:** Degradación Significativa (-{abs(anomaly):.1f}%)")
                st.markdown("**Diagnóstico:** El valor actual está muy por debajo del promedio histórico. Posible sequía o sobrepastoreo.")
            elif anomaly > 15:
                st.success(f"🌱 **CONDICIÓN FAVORABLE:** Superávit de Biomasa (+{anomaly:.1f}%)")
                st.markdown("**Diagnóstico:** Condiciones superiores al promedio histórico.")
            else:
                st.info(f"⚖️ **ESTABLE:** Variación Normal ({anomaly:.1f}%)")
                st.markdown("**Diagnóstico:** Los valores se mantienen dentro del rango esperado.")
        
        with col_metric:
            st.metric("Promedio Histórico", f"{hist_mean:.2f}", f"{anomaly:.1f}% vs Promedio")

st.markdown('</div>', unsafe_allow_html=True)

# FIRMA
st.markdown('<div class="footer">Jhon Monroy Experto en informatica</div>', unsafe_allow_html=True)
