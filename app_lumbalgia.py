import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import datetime

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Salud Lumbar IA", layout="wide")

# --- 2. FUNCIÓN PARA GENERAR EL REPORTE PDF (CORREGIDA) ---
def crear_pdf(nombre, edad, imc, riesgo, prob, agua, proteina, huesos, consejos):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado profesional
    pdf.set_fill_color(26, 82, 118)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 25, 'INFORME DE SALUD LUMBAR - IA', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, f'PACIENTE: {nombre.upper()}', 0, 1, 'L')
    pdf.cell(190, 10, f'EDAD: {edad} años | FECHA: {datetime.date.today()}', 0, 1, 'L')
    pdf.ln(5)

    # Tabla de métricas
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
    pdf.cell(190, 15, f'DIAGNOSTICO DE RIESGO: {riesgo} (Confianza: {prob:.1f}%)', 1, 1, 'C')
    
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'RECOMENDACIONES Y PRIORIDAD:', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    for c in consejos:
        # Limpieza de caracteres especiales para evitar error Unicode en PDF
        c_limpia = c.encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(190, 8, f'- {c_limpia}')
        
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# --- 3. CARGA Y ENTRENAMIENTO (CON TU LÓGICA DE LOCALIZACIÓN) ---
@st.cache_data(ttl=600)
def entrenar_modelo_dinamico():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0W8wtAAKG-qj0BppHi_Qu_LtuvjQ5pOCYDYQdRdwD01mCSjIH8tLn3-KyP8OnrYVEWXV2O4rrVmx/pub?gid=2099638101&single=true&output=csv"
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]

    def localizar(keyword):
        for c in df.columns:
            if keyword in c: return c
        return None

    # Mapeo de columnas reales
    c_peso, c_talla = localizar('peso'), localizar('estatura')
    c_estres, c_genero = localizar('estres'), localizar('genero')
    c_pausas, c_horas = localizar('pausas'), localizar('horas')
    c_silla, c_target = localizar('silla'), localizar('dolor')

    # Procesamiento
    df['imc_calc'] = pd.to_numeric(df[c_peso], errors='coerce') / ((pd.to_numeric(df[c_talla], errors='coerce')/100)**2)
    df['nivel_estres'] = pd.to_numeric(df[c_estres], errors='coerce').fillna(3)
    df['genero_num'] = df[c_genero].map({'Masculino': 0, 'Femenino': 1, 'Otro': 2}).fillna(0)
    df['pausas_num'] = df[c_pausas].map({'Nunca': 0, 'Ocasionalmente': 1, 'Diariamente': 2}).fillna(1)
    df['horas_num'] = df[c_horas].map({'1-4': 1, '4-6': 2, '6-8': 3, 'Más de 8': 4}).fillna(2)
    df['silla_num'] = df[c_silla].apply(lambda x: 1 if 'si' in str(x).lower() else 0)
    df['label'] = df[c_target].apply(lambda x: 1 if 'si' in str(x).lower() else 0)

    df_clean = df.dropna(subset=['imc_calc', 'nivel_estres', 'label'])
    features = ['imc_calc', 'nivel_estres', 'genero_num', 'pausas_num', 'horas_num', 'silla_num']
    
    modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo.fit(df_clean[features], df_clean['label'])
    return modelo, len(df)

# --- 4. INTERFAZ Y LÓGICA DE SALIDA ---
try:
    modelo_ia, total_reg = entrenar_modelo_dinamico()
except:
    st.error("Error al conectar con la base de datos.")
    st.stop()

st.title("🏥 Asistente de Salud Lumbar - IA Multivariable")

with st.sidebar:
    st.info(f"📈 IA entrenada con {total_reg} registros.")
    nombre_user = st.text_input("Nombre completo", "Colaborador")
    genero_in = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
    edad_user = st.number_input("Edad", 18, 90, 33)
    peso_user = st.number_input("Peso Actual (kg)", 30.0, 200.0, 70.0)
    talla_user = st.number_input("Estatura (cm)", 100, 250, 170)
    estres_user = st.slider("Nivel de Estrés (1-5)", 1, 5, 3)
    horas_in = st.selectbox("Horas sentado al día", ["1-4", "4-6", "6-8", "Más de 8"])
    pausas_in = st.selectbox("Frecuencia de pausas activas", ["Nunca", "Ocasionalmente", "Diariamente"])
    silla_in = st.radio("¿Silla con soporte lumbar?", ["Sí", "No"])

# Conversiones
imc_user = peso_user / ((talla_user/100)**2)
gen_num = {"Masculino": 0, "Femenino": 1, "Otro": 2}[genero_in]
hor_num = {"1-4": 1, "4-6": 2, "6-8": 3, "Más de 8": 4}[horas_in]
pau_num = {"Nunca": 0, "Ocasionalmente": 1, "Diariamente": 2}[pausas_in]
sil_num = 1 if silla_in == "Sí" else 0

# Dashboard de métricas
agua, prot, hueso = peso_user * 0.55, peso_user * 0.17, peso_user * 0.04
c1, c2, c3, c4 = st.columns(4)
c1.metric("IMC", f"{imc_user:.2f}")
c2.metric("Agua", f"{agua:.1f}kg")
c3.metric("Proteína", f"{prot:.1f}kg")
c4.metric("Huesos", f"{hueso:.1f}kg")

if st.button("🔍 ANALIZAR RIESGO Y GENERAR REPORTE"):
    input_data = [[imc_user, estres_user, gen_num, pau_num, hor_num, sil_num]]
    pred = modelo_ia.predict(input_data)[0]
    prob = max(modelo_ia.predict_proba(input_data)[0]) * 100
    riesgo = "ALTO" if pred == 1 else "BAJO"
    
    consejos = []
    if riesgo == "ALTO":
        # Implementación de la escala de tiempo/urgencia
        if estres_user >= 4 or hor_num >= 3:
            urgencia = "ALTA PRIORIDAD: Intervencion necesaria en los proximos 30 dias."
            st.error(f"### RIESGO {riesgo}")
            st.error(f"🔴 {urgencia}")
        else:
            urgencia = "PRIORIDAD MEDIA: Se recomienda realizar ajustes en los proximos 3 meses."
            st.error(f"### RIESGO {riesgo}")
            st.warning(f"🟠 {urgencia}")
        consejos.append(urgencia)
    else:
        st.success(f"### RIESGO {riesgo}")
        consejos.append("MANTENIMIENTO: Siga sus habitos actuales y realice pausas.")

    # Recomendaciones extra
    if sil_num == 0: consejos.append("Adquirir un soporte lumbar para mejorar la postura.")
    if imc_user > 25: consejos.append("Se recomienda valoracion nutricional por IMC elevado.")
    
    st.write(f"**Confianza del modelo:** {prob:.1f}%")
    
    pdf_bytes = crear_pdf(nombre_user, edad_user, imc_user, riesgo, prob, agua, prot, hueso, consejos)
    st.download_button("📄 Descargar Informe PDF", pdf_bytes, f"Reporte_{nombre_user}.pdf", "application/pdf")