import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import datetime

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Health Analytics", layout="wide")

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
    pdf.cell(190, 10, 'RECOMENDACIONES MEDICAS:', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    for c in consejos:
        pdf.multi_cell(190, 8, f'- {c}')
        
    return pdf.output(dest='S').encode('latin-1')

# --- 3. ENTRENAMIENTO DEL MODELO (CRISP-DM) ---
@st.cache_data(ttl=600)
def entrenar_modelo():
    try:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0W8wtAAKG-qj0BppHi_Qu_LtuvjQ5pOCYDYQdRdwD01mCSjIH8tLn3-KyP8OnrYVEWXV2O4rrVmx/pub?gid=2099638101&single=true&output=csv"
        df = pd.read_csv(url)
        df = df.drop_duplicates()
        
        # Limpieza estándar: todo a minúsculas y sin espacios raros
        df.columns = [c.strip().lower() for c in df.columns]

        # --- FUNCIÓN INTELIGENTE PARA ENCONTRAR COLUMNAS ---
        def localizar(keyword):
            for c in df.columns:
                if keyword in c:
                    return c
            return None

        # Localizamos los nombres reales de las columnas en tu Sheets
        c_peso = localizar('peso')
        c_talla = localizar('estatura') or localizar('talla')
        c_estres = localizar('estres') or localizar('estrés')
        c_genero = localizar('género') or localizar('genero')
        c_pausas = localizar('pausas')
        c_horas = localizar('horas')
        c_silla = localizar('silla')
        c_target = localizar('dolor') or localizar('molestia')

        # --- PROCESAMIENTO DE DATOS ---
        df['imc_calc'] = pd.to_numeric(df[c_peso], errors='coerce') / ((pd.to_numeric(df[c_talla], errors='coerce')/100)**2)
        df['nivel_estres'] = pd.to_numeric(df[c_estres], errors='coerce').fillna(3)
        
        # Encodings con .get() para evitar errores si el texto varía un poco
        df['genero_num'] = df[c_genero].map({'Masculino': 0, 'Femenino': 1, 'Otro': 2}).fillna(0)
        df['pausas_num'] = df[c_pausas].map({'Nunca': 0, 'Ocasionalmente': 1, 'Diariamente': 2}).fillna(1)
        df['horas_num'] = df[c_horas].map({'1-4': 1, '4-6': 2, '6-8': 3, 'Más de 8': 4}).fillna(2)
        df['silla_num'] = df[c_silla].apply(lambda x: 1 if 'sí' in str(x).lower() or 'si' in str(x).lower() else 0)
        
        # Variable Objetivo (Target)
        df['label'] = df[c_target].apply(lambda x: 1 if 'sí' in str(x).lower() or 'si' in str(x).lower() else 0)
        
        # Limpieza de nulos antes de entrenar
        df_clean = df.dropna(subset=['imc_calc', 'nivel_estres', 'label'])
        
        features = ['imc_calc', 'nivel_estres', 'genero_num', 'pausas_num', 'horas_num', 'silla_num']
        X = df_clean[features]
        y = df_clean['label']
        
        modelo = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, class_weight='balanced', random_state=42)
        modelo.fit(X, y)
        return modelo, len(df)

    except Exception as e:
        st.error(f"Error detallado: {e}")
        # Esto te ayudará a ver cómo está leyendo Pandas las columnas si vuelve a fallar
        if 'df' in locals():
            st.write("Columnas que la IA encontró en tu Excel:", list(df.columns))
        return None, 0
    
modelo_ia, total_registros = entrenar_modelo()

# --- 4. INTERFAZ DE USUARIO ---
st.title("🏥 Asistente de Salud Lumbar - IA Multivariable")

if modelo_ia is not None:
    with st.sidebar:
        # A. ESTADO DEL ENTRENAMIENTO
        st.info(f"📈 IA entrenada con {total_registros} registros actuales.")
        st.divider()

        # B. INGRESO DE DATOS DEL USUARIO
        st.header("📋 Datos del Colaborador")
        nombre_user = st.text_input("Nombre completo", "su nombre aqui")
        genero_input = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
        edad_user = st.number_input("Edad", 18, 90, 33)
        peso_user = st.number_input("Peso Actual (kg)", 30.0, 200.0, 75.8)
        talla_user = st.number_input("Estatura (cm)", 100, 250, 169)
        
        st.divider()
        st.header("🏢 Entorno Laboral")
        estres_user = st.slider("Nivel de Estrés (1-5)", 1, 5, 3)
        horas_input = st.selectbox("Horas sentado al día", ["1-4", "4-6", "6-8", "Más de 8"])
        pausas_input = st.selectbox("Frecuencia de pausas activas", ["Nunca", "Ocasionalmente", "Diariamente"])
        silla_input = st.radio("¿Su silla tiene soporte lumbar?", ["Sí", "No"])

        # Mapeos internos
        genero_user = {"Masculino": 0, "Femenino": 1, "Otro": 2}[genero_input]
        horas_user = {"1-4": 1, "4-6": 2, "6-8": 3, "Más de 8": 4}[horas_input]
        pausas_user = {"Nunca": 0, "Ocasionalmente": 1, "Diariamente": 2}[pausas_input]
        silla_user = 1 if silla_input == "Sí" else 0

        st.divider()

        # C. SESIÓN DE COLABORACIÓN
        st.subheader("🤝 Colabora con la IA")
        st.write("Tu experiencia ayuda a mejorar la precisión. Al hacer clic, serás redirigido a un formulario para aportar tus datos al estudio.")
        st.link_button("Registrar mis datos en el estudio", 
                      "https://docs.google.com/forms/d/e/1FAIpQLScXVZ7iCF1nad67S8x2aNUK_c6v3hSRyTNT9K20m0MR-aETVg/viewform")
        st.caption("Nota: La recolección es 100% anónima.")

        st.divider()

        # D. CONTADOR DE VISITAS
        if 'visitas' not in st.session_state:
            st.session_state.visitas = 1
        else:
            st.session_state.visitas += 1
        st.metric(label="👥 Consultas en esta sesión", value=st.session_state.visitas)

    # --- DASHBOARD PRINCIPAL ---
    imc_user = peso_user / ((talla_user/100)**2)
    agua_user, prot_user, hueso_user = peso_user * 0.55, peso_user * 0.17, peso_user * 0.04

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IMC", f"{imc_user:.2f}")
    c2.metric("Agua 💧", f"{agua_user:.1f}kg")
    c3.metric("Proteína 🥩", f"{prot_user:.1f}kg")
    c4.metric("Huesos 🦴", f"{hueso_user:.1f}kg")

    st.divider()

    if st.button("🔍 ANALIZAR RIESGO Y GENERAR REPORTE"):
        datos_entrada = [[imc_user, estres_user, genero_user, pausas_user, horas_user, silla_user]]
        pred = modelo_ia.predict(datos_entrada)
        prob = max(modelo_ia.predict_proba(datos_entrada)[0]) * 100
        riesgo = "ALTO" if pred[0] == 1 else "BAJO"
        
        if riesgo == "ALTO":
            # Determinamos el nivel de urgencia dentro del riesgo alto
            if estres_user >= 4 or horas_user >= 3:
                texto_urgencia = "🔴 ALTA PRIORIDAD: Intervención necesaria en el corto plazo (próximos 30 días) para evitar cronicidad."
                st.error(f"### DIAGNÓSTICO: RIESGO {riesgo}")
                st.error(texto_urgencia) # Rojo para alta prioridad
            else:
                texto_urgencia = "🟠 PRIORIDAD MEDIA: Se recomienda realizar ajustes en los próximos 3 meses."
                st.error(f"### DIAGNÓSTICO: RIESGO {riesgo}")
                st.warning(texto_urgencia) # Naranja para prioridad media
            
            recomendaciones.append(texto_urgencia)
        else:
            # Para riesgo bajo, no hay "urgencia", hay "mantenimiento"
            st.success(f"### DIAGNÓSTICO: RIESGO {riesgo}")
            texto_mantenimiento = "🟢 MANTENIMIENTO: Siga sus hábitos actuales y realice chequeos preventivos semestrales."
            st.info(texto_mantenimiento)
            recomendaciones.append(texto_mantenimiento)

        # Recomendaciones específicas
        recomendaciones.append("Realizar pausas activas cada 45 minutos (estiramientos de cadena posterior).")
        if silla_user == 0: recomendaciones.append("Adquirir un soporte lumbar o cojín ergonómico para corregir la lordosis.")
        if imc_user > 25: recomendaciones.append("Valoración nutricional: el exceso de peso aumenta la presión intradiscal.")

        pdf_data = crear_pdf(nombre_user, edad_user, imc_user, riesgo, prob, agua_user, prot_user, hueso_user, recomendaciones)
        st.download_button(label="📄 Descargar Informe PDF", data=pdf_data, file_name=f"Salud_{nombre_user}.pdf", mime="application/pdf")