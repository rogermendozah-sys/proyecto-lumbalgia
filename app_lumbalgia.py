import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import datetime

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Health Analytics - Roger Mendoza", layout="wide")

# --- 2. FUNCIÓN PARA GENERAR EL REPORTE PDF ---
def crear_pdf(nombre, edad, imc, riesgo, prob, agua, proteina, huesos, consejos):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado azul profesional
    pdf.set_fill_color(26, 82, 118)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 25, 'INFORME DE SALUD LUMBAR - IA', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, f'PACIENTE: {nombre}', 0, 1, 'L')
    pdf.cell(190, 10, f'EDAD: {edad} años | FECHA: {datetime.date.today()}', 0, 1, 'L')
    pdf.ln(5)

    # Tabla de métricas (Como la báscula)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(47, 10, 'IMC', 1, 0, 'C', True)
    pdf.cell(47, 10, 'AGUA (kg)', 1, 0, 'C', True)
    pdf.cell(47, 10, 'PROTEINA (kg)', 1, 0, 'C', True)
    pdf.cell(49, 10, 'HUESOS (kg)', 1, 1, 'C', True)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(47, 10, f'{imc:.2f}', 1, 0, 'C')
    pdf.cell(47, 10, f'{agua:.1f}', 1, 0, 'C')
    pdf.cell(47, 10, f'{proteina:.1f}', 1, 0, 'C')
    pdf.cell(49, 10, f'{huesos:.1f}', 1, 1, 'C')
    
    pdf.ln(10)
    
    # Resultado de la IA
    pdf.set_font('Arial', 'B', 14)
    color = (200, 0, 0) if riesgo == "ALTO" else (0, 150, 0)
    pdf.set_text_color(*color)
    pdf.cell(190, 15, f'DIAGNOSTICO DE RIESGO: {riesgo}', 1, 1, 'C')
    
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'RECOMENDACIONES MEDICAS:', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    for c in consejos:
        pdf.multi_cell(190, 8, f'- {c}')
        
    return pdf.output(dest='S').encode('latin-1')

# --- 3. ENTRENAMIENTO DEL MODELO (IA) ---
@st.cache_data
def entrenar_modelo():
    try:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0W8wtAAKG-qj0BppHi_Qu_LtuvjQ5pOCYDYQdRdwD01mCSjIH8tLn3-KyP8OnrYVEWXV2O4rrVmx/pub?gid=2099638101&single=true&output=csv"
        df = pd.read_csv(url)
        
        # Limpiar nombres de columnas
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Identificar columnas por palabras clave
        col_peso = [c for c in df.columns if 'peso' in c][0]
        col_talla = [c for c in df.columns if 'estatura' in c or 'talla' in c or 'cms' in c][0]
        col_estres = [c for c in df.columns if 'estres' in c or 'estrés' in c][0]
        col_target = [c for c in df.columns if 'dolor' in c or 'target' in c][0]

        # Crear variables calculadas
        df['imc_calc'] = pd.to_numeric(df[col_peso], errors='coerce') / ((pd.to_numeric(df[col_talla], errors='coerce')/100)**2)
        df['nivel_estres'] = pd.to_numeric(df[col_estres], errors='coerce').fillna(3)
        df['label'] = df[col_target].apply(lambda x: 1 if 'sí' in str(x).lower() or 'si' in str(x).lower() else 0)
        
        # Entrenar modelo
        df_clean = df.dropna(subset=['imc_calc', 'nivel_estres', 'label'])
        X = df_clean[['imc_calc', 'nivel_estres']]
        y = df_clean['label']
        
        modelo = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
        modelo.fit(X, y)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Inicializar el modelo globalmente
modelo_ia = entrenar_modelo()

if modelo_ia is not None:
    # Usamos la misma URL que ya tienes arriba
    url_datos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0W8wtAAKG-qj0BppHi_Qu_LtuvjQ5pOCYDYQdRdwD01mCSjIH8tLn3-KyP8OnrYVEWXV2O4rrVmx/pub?gid=2099638101&single=true&output=csv"
    df_temp = pd.read_csv(url_datos)
    total_registros = len(df_temp)
    
    # Esto lo muestra en la barra lateral de forma elegante
    st.sidebar.info(f"📈 IA entrenada con {total_registros} registros actuales.")
# ----------------------------------------------

# --- 4. INTERFAZ DE USUARIO (DASHBOARD) ---
st.title("🏥 Asistente de Salud Lumbar - IA")

if modelo_ia is not None:
    with st.sidebar:
        st.header("📋 Ingreso de Datos")
        nombre_user = st.text_input("Nombre del Colaborador", "Roger_DMendoza")
        edad_user = st.number_input("Edad", 18, 90, 33)
        peso_user = st.number_input("Peso Actual (kg)", 30.0, 200.0, 75.8)
        talla_user = st.number_input("Estatura (cm)", 100, 250, 169)
        estres_user = st.slider("Nivel de Estrés (1-5)", 1, 5, 3)

    # Cálculos de composición corporal (Simulación de báscula)
    imc_user = peso_user / ((talla_user/100)**2)
    agua_user = peso_user * 0.55
    prot_user = peso_user * 0.17
    hueso_user = peso_user * 0.04

    # Mostrar Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IMC", f"{imc_user:.2f}")
    col2.metric("Agua 💧", f"{agua_user:.1f}kg")
    col3.metric("Proteína 🥩", f"{prot_user:.1f}kg")
    col4.metric("Huesos 🦴", f"{hueso_user:.1f}kg")

    st.divider()

    # Botón de Diagnóstico
    if st.button("🔍 ANALIZAR RIESGO Y GENERAR PDF"):
        # Predicción
        prediccion = modelo_ia.predict([[imc_user, estres_user]])
        probabilidad = modelo_ia.predict_proba([[imc_user, estres_user]])
        
        riesgo_final = "ALTO" if prediccion[0] == 1 else "BAJO"
        confianza = max(probabilidad[0]) * 100
        
        # Mostrar resultado en pantalla
        if riesgo_final == "ALTO":
            st.error(f"### RESULTADO: RIESGO {riesgo_final}")
        else:
            st.success(f"### RESULTADO: RIESGO {riesgo_final}")
            
        st.info(f"Confianza del algoritmo: {confianza:.1f}%")

        # Recomendaciones
        recomendaciones = [
            "Realizar pausas activas cada 45 minutos.",
            "Ajustar la altura del monitor a la línea de los ojos.",
            "Utilizar soporte lumbar en la silla de oficina."
        ]
        if imc_user > 25:
            recomendaciones.append("Se sugiere valoración nutricional para control de peso.")

        # Generar y habilitar descarga de PDF
        pdf_data = crear_pdf(nombre_user, edad_user, imc_user, riesgo_final, confianza, agua_user, prot_user, hueso_user, recomendaciones)
        
        st.download_button(
            label="📄 Descargar Informe PDF",
            data=pdf_data,
            file_name=f"Informe_Salud_{nombre_user}.pdf",
            mime="application/pdf"
        )
else:
    st.warning("No se pudo cargar el modelo. Verifica tu conexión o el archivo de datos.")