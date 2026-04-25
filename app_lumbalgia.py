import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import io

# Configuración de página
st.set_page_config(page_title="Asistente Salud Lumbar", page_icon="🏥", layout="wide")

# --- FUNCIONES DE DATOS ---
@st.cache_data(ttl=600) # Se actualiza cada 10 minutos
def cargar_datos():
    # El link que me pasaste configurado para lectura directa
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0W8wtAAKG-qj0BppHi_Qu_LtuvjQ5pOCYDYQdRdwD01mCSjIH8tLn3-KyP8OnrYVEWXV2O4rrVmx/pub?gid=2099638101&single=true&output=csv"
    df = pd.read_csv(sheet_url)
    return df

def entrenar_modelo(df):
    # Definimos las variables predictoras (Asegúrate que estos nombres coincidan con tu CSV)
    # Si tu CSV tiene nombres distintos, cámbiados aquí:
    features = ['imc_calc', 'nivel_estres', 'genero_num', 'pausas_num', 'horas_num', 'silla_num']
    
    X = df[features]
    y = df['dolor_lumbar_num'] # 1: Sí, 0: No
    
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
    
    # Encabezado
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="REPORTE DE RIESGO LUMBAR - IA", ln=True, align='C')
    pdf.ln(5)
    
    # Datos personales
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Nombre: {nombre}", ln=True)
    pdf.cell(200, 10, txt=f"Edad: {edad} años", ln=True)
    pdf.cell(200, 10, txt=f"IMC: {imc:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Resultado: RIESGO {riesgo}", ln=True)
    pdf.cell(200, 10, txt=f"Nivel de Confianza: {prob:.1f}%", ln=True)
    
    # Sección de Recomendaciones
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="PLAN DE ACCIÓN RECOMENDADO:", ln=True)
    pdf.set_font("Arial", size=11)
    
    for rec in recomendaciones:
        # Limpiamos emojis para evitar errores de codificación en PDF
        rec_limpia = rec.replace("🔴 ", "").replace("🟠 ", "").replace("🟢 ", "")
        pdf.multi_cell(0, 8, txt=f"- {rec_limpia}")
        
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# --- INTERFAZ ---
st.title("🏥 Asistente de Salud Lumbar - IA Multivariable")

try:
    data = cargar_datos()
    modelo = entrenar_modelo(data)
    st.sidebar.info(f"✅ IA entrenada con {len(data)} registros en tiempo real.")
except Exception as e:
    st.sidebar.error("⚠️ Error conectando con Google Sheets. Verifique los nombres de las columnas.")
    st.stop()

# --- SIDEBAR: ENTRADA DE DATOS ---
with st.sidebar:
    st.header("📋 Datos del Usuario")
    nombre_user = st.text_input("Nombre completo", "Colaborador")
    genero = st.selectbox("Género", ["Masculino", "Femenino"])
    edad_user = st.number_input("Edad", 18, 80, 33)
    peso = st.number_input("Peso (kg)", 40.0, 150.0, 70.0)
    estatura = st.number_input("Estatura (cm)", 100, 250, 170)
    
    st.header("🏢 Entorno Laboral")
    estres_user = st.slider("Nivel de Estrés Percibido (1-5)", 1, 5, 3)
    horas_opcion = st.selectbox("Horas sentado al día", ["1-4", "4-8", "Más de 8"])
    pausas_opcion = st.selectbox("Frecuencia de pausas", ["Nunca", "Ocasionalmente", "Diariamente"])
    silla_opcion = st.radio("¿Silla ergonómica?", ["Sí", "No"])

# Conversión para el modelo
imc_user = peso / ((estatura/100)**2)
genero_num = 1 if genero == "Masculino" else 0
pausas_num = 2 if pausas_opcion == "Diariamente" else (1 if pausas_opcion == "Ocasionalmente" else 0)
horas_num = 1 if horas_opcion == "1-4" else (2 if horas_opcion == "4-8" else 3)
silla_num = 1 if silla_opcion == "Sí" else 0

# --- CÁLCULO DE RESULTADOS ---
if st.button("🔍 GENERAR DIAGNÓSTICO"):
    # Preparar datos para predicción
    X_input = [[imc_user, estres_user, genero_num, pausas_num, horas_num, silla_num]]
    pred = modelo.predict(X_input)[0]
    prob = modelo.predict_proba(X_input)[0][pred] * 100
    riesgo = "ALTO" if pred == 1 else "BAJO"

    recomendaciones = []
    
    # Lógica de prioridad y mensajes
    if riesgo == "ALTO":
        if estres_user >= 4 or horas_num == 3:
            msg = "ALTA PRIORIDAD: Intervención necesaria en los próximos 30 días."
            st.error(f"### ⚠️ DIAGNÓSTICO: RIESGO {riesgo}")
            st.error(f"🔴 {msg}")
            recomendaciones.append(f"🔴 {msg}")
        else:
            msg = "PRIORIDAD MEDIA: Se recomienda ajustes en los próximos 3 meses."
            st.error(f"### ⚠️ DIAGNÓSTICO: RIESGO {riesgo}")
            st.warning(f"🟠 {msg}")
            recomendaciones.append(f"🟠 {msg}")
    else:
        st.success(f"### ✅ DIAGNÓSTICO: RIESGO {riesgo}")
        msg = "MANTENIMIENTO: Continúe con sus hábitos preventivos."
        st.info(f"🟢 {msg}")
        recomendaciones.append(f"🟢 {msg}")

    # Recomendaciones dinámicas adicionales
    recomendaciones.append("Realizar pausas activas con estiramientos cada 45-60 min.")
    if silla_num == 0: recomendaciones.append("Es prioritario el uso de una silla con soporte lumbar.")
    if imc_user > 25: recomendaciones.append("Controlar el peso corporal para reducir presión en discos intervertebrales.")

    st.write(f"**Nivel de confianza del modelo:** {prob:.1f}%")

    # Botón de Descarga
    pdf_bytes = crear_pdf(nombre_user, edad_user, imc_user, riesgo, prob, recomendaciones)
    st.download_button(
        label="📄 Descargar Informe Completo (PDF)",
        data=pdf_bytes,
        file_name=f"Resultado_IA_{nombre_user}.pdf",
        mime="application/pdf"
    )