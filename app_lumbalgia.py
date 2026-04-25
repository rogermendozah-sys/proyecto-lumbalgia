import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import io

# Configuración de página
st.set_page_config(page_title="Asistente Salud Lumbar", page_icon="🏥")

# --- FUNCIONES DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos():
    # Reemplaza con tu URL real de Google Sheets (formato export?format=csv)
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0W8wtAAKG-qj0BppHi_Qu_LtuvjQ5pOCYDYQdRdwD01mCSjIH8tLn3-KyP8OnrYVEWXV2O4rrVmx/pubhtml?gid=2099638101&single=true"
    df = pd.read_csv(sheet_url)
    return df

def entrenar_modelo(df):
    # Asumiendo que tus columnas numéricas ya están preparadas en el CSV
    # Ajusta los nombres de las columnas según tu archivo real
    features = ['imc_calc', 'nivel_estres', 'genero_num', 'pausas_num', 'horas_num', 'silla_num']
    X = df[features]
    y = df['dolor_lumbar_num'] # 1 si tiene dolor, 0 si no
    
    modelo = DecisionTreeClassifier(
        max_depth=3, 
        min_samples_leaf=10, 
        class_weight='balanced', 
        random_state=42
    )
    modelo.fit(X, y)
    return modelo

# --- FUNCIÓN GENERAR PDF ---
def crear_pdf(nombre, edad, imc, riesgo, prob, recomendaciones):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte de Riesgo Lumbar - IA", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Nombre: {nombre}", ln=True)
    pdf.cell(200, 10, txt=f"Edad: {edad} años", ln=True)
    pdf.cell(200, 10, txt=f"IMC: {imc:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Diagnóstico: RIESGO {riesgo}", ln=True)
    pdf.cell(200, 10, txt=f"Confianza de la IA: {prob:.1f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Recomendaciones y Plan de Acción:", ln=True)
    
    pdf.set_font("Arial", size=11)
    for rec in recomendaciones:
        # Eliminamos caracteres especiales que rompen el PDF
        rec_limpia = rec.replace("🔴 ", "").replace("🟠 ", "").replace("🟢 ", "")
        pdf.multi_cell(0, 10, txt=f"- {rec_limpia}")
        
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# --- INTERFAZ STREAMLIT ---
st.title("🏥 Asistente de Salud Lumbar - IA Multivariable")

# Intentar cargar datos y entrenar
try:
    df = cargar_datos()
    modelo = entrenar_modelo(df)
    st.sidebar.success(f"IA entrenada con {len(df)} registros.")
except:
    st.sidebar.error("Conectando con base de datos...")
    st.stop()

# Formulario lateral
with st.sidebar:
    st.header("📋 Datos del Colaborador")
    nombre_user = st.text_input("Nombre completo", "Usuario")
    genero = st.selectbox("Género", ["Masculino", "Femenino"])
    edad_user = st.number_input("Edad", 18, 80, 33)
    peso = st.number_input("Peso Actual (kg)", 40.0, 150.0, 70.0)
    estatura = st.number_input("Estatura (cm)", 100, 250, 170)
    
    st.header("🏢 Entorno Laboral")
    estres_user = st.slider("Nivel de Estrés (1-5)", 1, 5, 3)
    horas_user = st.selectbox("Horas sentado al día", ["1-4", "4-8", "Más de 8"])
    pausas_user = st.selectbox("Frecuencia de pausas activas", ["Nunca", "Ocasionalmente", "Diariamente"])
    silla_user_text = st.radio("¿Su silla tiene soporte lumbar?", ["Sí", "No"])

# Procesamiento de variables para la IA
genero_num = 1 if genero == "Masculino" else 0
imc_user = peso / ((estatura/100)**2)
pausas_num = 2 if pausas_user == "Diariamente" else (1 if pausas_user == "Ocasionalmente" else 0)
horas_num = 1 if horas_user == "1-4" else (2 if horas_user == "4-8" else 3)
silla_num = 1 if silla_user_text == "Sí" else 0

# --- BOTÓN DE ANÁLISIS ---
if st.button("🔍 ANALIZAR RIESGO Y GENERAR REPORTE"):
    # Predicción
    datos_entrada = [[imc_user, estres_user, genero_num, pausas_num, horas_num, silla_num]]
    riesgo_idx = modelo.predict(datos_entrada)[0]
    probabilidades = modelo.predict_proba(datos_entrada)[0]
    
    riesgo = "ALTO" if riesgo_idx == 1 else "BAJO"
    confianza = probabilidades[riesgo_idx] * 100

    # Lógica de Recomendaciones y Visualización
    recomendaciones = []
    
    if riesgo == "ALTO":
        if estres_user >= 4 or horas_num >= 3:
            msg = "ALTA PRIORIDAD: Intervención necesaria en el corto plazo (próximos 30 días)."
            st.error(f"### DIAGNÓSTICO: RIESGO {riesgo}")
            st.error(f"🔴 {msg}")
            recomendaciones.append(f"🔴 {msg}")
        else:
            msg = "PRIORIDAD MEDIA: Se recomienda realizar ajustes en los próximos 3 meses."
            st.error(f"### DIAGNÓSTICO: RIESGO {riesgo}")
            st.warning(f"🟠 {msg}")
            recomendaciones.append(f"🟠 {msg}")
    else:
        st.success(f"### DIAGNÓSTICO: RIESGO {riesgo}")
        msg = "MANTENIMIENTO: Siga sus hábitos actuales y realice chequeos preventivos."
        st.info(f"🟢 {msg}")
        recomendaciones.append(f"🟢 {msg}")

    # Recomendaciones extra
    recomendaciones.append("Realizar pausas activas cada 45 minutos.")
    if silla_num == 0: recomendaciones.append("Usar soporte lumbar ergonómico.")
    if imc_user > 25: recomendaciones.append("Valoración nutricional por sobrepeso.")

    st.write(f"**Confianza del modelo:** {confianza:.1f}%")

    # Botón PDF
    pdf_bytes = crear_pdf(nombre_user, edad_user, imc_user, riesgo, confianza, recomendaciones)
    st.download_button(
        label="📄 Descargar Informe PDF",
        data=pdf_bytes,
        file_name=f"Reporte_Lumbar_{nombre_user}.pdf",
        mime="application/pdf"
    )