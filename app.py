%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="IA Deserción UTP", page_icon="🎓", layout="centered")

# --- CARGAR MODELO (Rutas simplificadas) ---
@st.cache_resource
def cargar_inteligencia():
    # Buscamos los archivos en la misma carpeta donde corre la app
    if os.path.exists('modelo_desercion_utp.keras') and os.path.exists('escalador_utp.pkl'):
        model = tf.keras.models.load_model('modelo_desercion_utp.keras')
        scaler = joblib.load('escalador_utp.pkl')
        return model, scaler
    return None, None

st.title("🎓 Asistente IA: Diagnóstico de Deserción UTP")
st.markdown("Esta herramienta utiliza una **Red Neuronal Artificial** con 95.5% de precisión.")

# Cargar
model, scaler = cargar_inteligencia()

if model is None:
    st.error("⚠️ Error Crítico: No encuentro los archivos .keras o .pkl en la carpeta principal.")
    st.stop()

# --- INPUTS ---
st.sidebar.header("📝 Datos del Estudiante")
promedio = st.sidebar.slider("Promedio General", 0.0, 10.0, 8.0, step=0.1)
edad = st.sidebar.number_input("Edad", min_value=17, max_value=60, value=22)
cantidad_becas = st.sidebar.number_input("Becas Acumuladas", min_value=0, max_value=20, value=0)
genero_txt = st.sidebar.selectbox("Género", ["Femenino", "Masculino"])
residencia_txt = st.sidebar.selectbox("Residencia", ["Puebla (Local)", "Foráneo (Otro Estado)"])
beca_actual_txt = st.sidebar.selectbox("¿Ha tenido beca alguna vez?", ["Sí", "No"])

# --- PROCESAMIENTO ---
genero_val = 1 if genero_txt == "Masculino" else 0
foraneo_val = 1 if "Foráneo" in residencia_txt else 0
beca_val = 1 if beca_actual_txt == "Sí" else 0

input_data = pd.DataFrame([[promedio, edad, genero_val, foraneo_val, cantidad_becas, beca_val]],
                          columns=['PROMEDIO', 'EDAD', 'GENERO', 'ES_FORANEO', 'CANTIDAD_BECAS', 'TUVO_BECA'])

# Escalar y Predecir
if st.button("🔍 Realizar Diagnóstico", type="primary"):
    input_scaled = scaler.transform(input_data)
    prediction_prob = model.predict(input_scaled)[0][0]
    porcentaje = prediction_prob * 100

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        if porcentaje > 50:
            st.error("RIESGO ALTO")
        else:
            st.success("RIESGO BAJO")

    with col2:
        st.metric("Probabilidad de Deserción", f"{porcentaje:.2f}%")
        st.progress(int(porcentaje))
        if porcentaje > 50:
            st.warning("Recomendación: Canalizar a tutorías.")
        else:
            st.info("Recomendación: Trayectoria estable.")