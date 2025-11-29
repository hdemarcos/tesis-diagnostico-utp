import os
# Forzamos a utilizar la CPU para evitar errores de CUDA en la nube
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Deserción UTP", page_icon="🎓", layout="centered")

# --- CARGAR MODELO Y ESCALADOR ---
@st.cache_resource
def cargar_inteligencia():
    # En la nube, los archivos están en la misma carpeta, así que se llaman directo
    if os.path.exists('modelo_desercion_utp.keras') and os.path.exists('escalador_utp.pkl'):
        try:
            model = tf.keras.models.load_model('modelo_desercion_utp.keras')
            scaler = joblib.load('escalador_utp.pkl')
            return model, scaler
        except Exception as e:
            return None, None
    return None, None

model, scaler = cargar_inteligencia()

# --- INTERFAZ GRÁFICA ---
st.title("🎓 Diagnóstico de Deserción")
st.markdown("Sistema de Inteligencia Artificial - Universidad Tecnológica de Puebla")

if model is None:
    st.error("⚠️ Error de Sistema: No se encuentran los archivos del modelo (.keras o .pkl).")
    st.info("Verifica que hayas subido 'modelo_desercion_utp.keras' y 'escalador_utp.pkl' al repositorio de GitHub.")
    st.stop()

# --- FORMULARIO DE DATOS ---
st.markdown("---")
st.markdown("### 📝 Perfil del Estudiante")

# FILA 1
col1, col2 = st.columns(2)
with col1:
    st.info("📚 Desempeño Académico")
    promedio = st.slider("Promedio General", 0.0, 10.0, 8.5, step=0.1)
with col2:
    st.info("💰 Apoyos Económicos")
    cantidad_becas = st.number_input("Becas Acumuladas (Total)", 0, 15, 1)
    beca_actual_txt = st.selectbox("¿Ha tenido Beca alguna vez?", ["Sí", "No"])

# FILA 2
col3, col4 = st.columns(2)
with col3:
    st.warning("👤 Datos Personales")
    edad = st.number_input("Edad", 17, 60, 20)
with col4:
    st.warning("📍 Ubicación")
    genero_txt = st.selectbox("Género", ["Femenino", "Masculino"])
    residencia_txt = st.selectbox("Residencia", ["Puebla (Local)", "Foráneo"])

st.markdown("---")

# --- BOTÓN DE DIAGNÓSTICO ---
if st.button("🔍 CALCULAR RIESGO DE DESERCIÓN", type="primary", use_container_width=True):
    
    # Preprocesamiento de variables
    gen_val = 1 if genero_txt == "Masculino" else 0
    for_val = 1 if "Foráneo" in residencia_txt else 0
    beca_val = 1 if beca_actual_txt == "Sí" else 0
    
    # Crear DataFrame con el formato exacto que aprendió la IA
    input_data = pd.DataFrame([[promedio, edad, gen_val, for_val, cantidad_becas, beca_val]],
                          columns=['PROMEDIO', 'EDAD', 'GENERO', 'ES_FORANEO', 'CANTIDAD_BECAS', 'TUVO_BECA'])
    
    # Predicción
    try:
        input_scaled = scaler.transform(input_data)
        prediction_prob = model.predict(input_scaled)[0][0]
        porcentaje = prediction_prob * 100
        
        # Mostrar Resultados
        st.success("✅ Diagnóstico Completado")
        
        col_res_A, col_res_B = st.columns([1, 2])
        
        with col_res_A:
            st.metric("Probabilidad Calculada", f"{porcentaje:.2f}%")
        
        with col_res_B:
            st.write("### Nivel de Riesgo:")
            if porcentaje > 50:
                st.error(f"🔴 ALTO RIESGO ({porcentaje:.1f}%)")
                st.write("⚠️ **Recomendación:** Canalizar a Tutorías inmediatamente.")
            else:
                st.success(f"🟢 BAJO RIESGO ({porcentaje:.1f}%)")
                st.write("✅ **Recomendación:** Mantener seguimiento normal.")
            
            st.progress(int(porcentaje))
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar los datos: {e}")
