import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS PERSONALIZADOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuente y fondo general */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo principal */
    .stApp {
        background-color: #F0F4F8;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A2B45;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #3B82F6;
    }

    /* Tarjetas de sección */
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Encabezado principal */
    .main-header {
        background: linear-gradient(135deg, #1A2B45 0%, #2563EB 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 28px;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 6px 0 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }

    /* Badge de resultado */
    .result-high {
        background: #FEE2E2;
        border-left: 5px solid #DC2626;
        border-radius: 8px;
        padding: 20px 24px;
        color: #7F1D1D;
    }
    .result-low {
        background: #D1FAE5;
        border-left: 5px solid #059669;
        border-radius: 8px;
        padding: 20px 24px;
        color: #064E3B;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-prob {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 600;
    }

    /* Etiquetas de sección en sidebar */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #93C5FD !important;
        margin-top: 18px;
        margin-bottom: 4px;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    /* Pill de variable seleccionada */
    .var-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        margin: 3px;
        font-weight: 500;
    }

    /* Métrica personalizada */
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1A2B45;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 28px;
        width: 100%;
        transition: all 0.2s;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8, #1E40AF);
        box-shadow: 0 4px 12px rgba(37,99,235,0.35);
        transform: translateY(-1px);
    }

    /* Ocultar footer de Streamlit */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """
    Carga tu modelo entrenado.
    Reemplaza esta función con la carga real de tu modelo.
    Ejemplo:
        import joblib
        return joblib.load("modelo_cardiovascular.pkl")
    """
    # ── PLACEHOLDER: reemplaza con tu modelo real ──
    class ModeloSimulado:
        def predict_proba(self, X):
            # Simulación basada en factores de riesgo reales
            score = 0.1
            if X["presion_sistolica"].values[0] > 140: score += 0.20
            if X["presion_diastolica"].values[0] > 90: score += 0.10
            if X["colesterol"].values[0] >= 2: score += 0.15
            if X["glucosa"].values[0] >= 2: score += 0.10
            if X["fuma"].values[0] == 1: score += 0.10
            if X["consume_alcohol"].values[0] == 1: score += 0.05
            if X["actividad_fisica"].values[0] == 0: score += 0.05
            if X["edad_anios"].values[0] > 55: score += 0.10
            imc = X["Indice_Masa_Corp"].values[0]
            if imc > 30: score += 0.08
            elif imc > 25: score += 0.04
            score = min(score, 0.97)
            return np.array([[1 - score, score]])

        def predict(self, X):
            prob = self.predict_proba(X)[0][1]
            return np.array([1 if prob >= 0.5 else 0])

    return ModeloSimulado()

modelo = cargar_modelo()


# ─────────────────────────────────────────────
# DEFINICIÓN DE VARIABLES
# ─────────────────────────────────────────────
TODAS_LAS_VARIABLES = [
    "genero", "estatura_cm", "peso_kg",
    "presion_sistolica", "presion_diastolica",
    "colesterol", "glucosa",
    "fuma", "consume_alcohol", "actividad_fisica",
    "edad_anios", "Indice_Masa_Corp"
]

VARIABLES_INFO = {
    "genero":               {"tipo": "categorica", "opciones": {0: "Femenino", 1: "Masculino"}, "default": 0, "emoji": "👤"},
    "estatura_cm":          {"tipo": "numerica",   "min": 100, "max": 220, "default": 165, "unidad": "cm",  "emoji": "📏"},
    "peso_kg":              {"tipo": "numerica",   "min": 30,  "max": 200, "default": 70,  "unidad": "kg",  "emoji": "⚖️"},
    "presion_sistolica":    {"tipo": "numerica",   "min": 80,  "max": 250, "default": 120, "unidad": "mmHg","emoji": "🩺"},
    "presion_diastolica":   {"tipo": "numerica",   "min": 40,  "max": 150, "default": 80,  "unidad": "mmHg","emoji": "🩺"},
    "colesterol":           {"tipo": "ordinal",    "opciones": {1: "Normal", 2: "Superior al normal", 3: "Muy superior al normal"}, "default": 1, "emoji": "🧪"},
    "glucosa":              {"tipo": "ordinal",    "opciones": {1: "Normal", 2: "Superior al normal", 3: "Muy superior al normal"}, "default": 1, "emoji": "🍬"},
    "fuma":                 {"tipo": "binaria",    "opciones": {0: "No", 1: "Sí"}, "default": 0, "emoji": "🚬"},
    "consume_alcohol":      {"tipo": "binaria",    "opciones": {0: "No", 1: "Sí"}, "default": 0, "emoji": "🍷"},
    "actividad_fisica":     {"tipo": "binaria",    "opciones": {0: "No", 1: "Sí"}, "default": 1, "emoji": "🏃"},
    "edad_anios":           {"tipo": "numerica",   "min": 1,   "max": 100, "default": 45,  "unidad": "años","emoji": "🎂"},
    "Indice_Masa_Corp":     {"tipo": "numerica",   "min": 10,  "max": 60,  "default": 25,  "unidad": "kg/m²","emoji": "📊"},
}

NOMBRES_DISPLAY = {
    "genero":             "Género",
    "estatura_cm":        "Estatura",
    "peso_kg":            "Peso",
    "presion_sistolica":  "Presión Sistólica",
    "presion_diastolica": "Presión Diastólica",
    "colesterol":         "Colesterol",
    "glucosa":            "Glucosa",
    "fuma":               "Fuma",
    "consume_alcohol":    "Consume Alcohol",
    "actividad_fisica":   "Actividad Física",
    "edad_anios":         "Edad",
    "Indice_Masa_Corp":   "Índice de Masa Corporal",
}


# ─────────────────────────────────────────────
# ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫀 Predictor de Riesgo Cardiovascular</h1>
    <p>Ingresa los datos clínicos del paciente para obtener una estimación de riesgo basada en el modelo entrenado.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — SELECCIÓN DE VARIABLES Y DATOS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    # ── ID del paciente ──
    st.markdown('<p class="section-label">Identificación</p>', unsafe_allow_html=True)
    id_paciente = st.text_input("ID del paciente", value="PAC-001", label_visibility="collapsed",
                                 placeholder="ID del paciente")

    # ── Selección de variables ──
    st.markdown('<p class="section-label">Variables a ingresar</p>', unsafe_allow_html=True)
    variables_seleccionadas = st.multiselect(
        "Selecciona las variables",
        options=TODAS_LAS_VARIABLES,
        default=TODAS_LAS_VARIABLES,
        format_func=lambda x: f"{VARIABLES_INFO[x]['emoji']} {NOMBRES_DISPLAY[x]}",
        label_visibility="collapsed"
    )

    if not variables_seleccionadas:
        st.warning("⚠️ Selecciona al menos una variable.")

    st.markdown("---")
    st.markdown('<p class="section-label">Sobre el modelo</p>', unsafe_allow_html=True)
    st.info(
        "**Variable objetivo:** `enfermedad_cardiovascular`\n\n"
        "Las variables no seleccionadas usarán sus valores por defecto para la predicción.",
        icon="ℹ️"
    )


# ─────────────────────────────────────────────
# ÁREA PRINCIPAL — FORMULARIO DE INGRESO
# ─────────────────────────────────────────────
col_form, col_result = st.columns([3, 2], gap="large")

valores_ingresados = {}

with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### 📋 Datos del paciente — `{id_paciente}`")

    if not variables_seleccionadas:
        st.info("Selecciona variables en el panel lateral para comenzar.")
    else:
        # Agrupar variables por categoría
        GRUPOS = {
            "🏥 Datos demográficos":     ["genero", "edad_anios"],
            "📐 Medidas corporales":     ["estatura_cm", "peso_kg", "Indice_Masa_Corp"],
            "💉 Signos vitales":         ["presion_sistolica", "presion_diastolica"],
            "🧪 Resultados de laboratorio": ["colesterol", "glucosa"],
            "🌿 Estilo de vida":         ["fuma", "consume_alcohol", "actividad_fisica"],
        }

        for grupo, vars_grupo in GRUPOS.items():
            vars_activas = [v for v in vars_grupo if v in variables_seleccionadas]
            if not vars_activas:
                continue

            st.markdown(f"**{grupo}**")
            cols = st.columns(min(len(vars_activas), 2))

            for i, var in enumerate(vars_activas):
                info = VARIABLES_INFO[var]
                col = cols[i % 2]
                label = f"{info['emoji']} {NOMBRES_DISPLAY[var]}"

                with col:
                    if info["tipo"] == "numerica":
                        val = st.number_input(
                            label,
                            min_value=float(info["min"]),
                            max_value=float(info["max"]),
                            value=float(info["default"]),
                            step=1.0,
                            help=f"Unidad: {info.get('unidad', '')} — Rango: [{info['min']} – {info['max']}]"
                        )
                    elif info["tipo"] in ("categorica", "binaria", "ordinal"):
                        opciones = info["opciones"]
                        keys = list(opciones.keys())
                        labels = list(opciones.values())
                        default_idx = keys.index(info["default"]) if info["default"] in keys else 0
                        sel = st.selectbox(label, options=keys, index=default_idx,
                                           format_func=lambda x, op=opciones: op[x])
                        val = sel

                    valores_ingresados[var] = val

            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)

    # Botón de predicción centrado
    predict_btn = st.button("🔍 Calcular predicción de riesgo", use_container_width=True)


# ─────────────────────────────────────────────
# PANEL DE RESULTADOS
# ─────────────────────────────────────────────
with col_result:

    # ── Resumen de variables ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Resumen de entrada")

    if valores_ingresados:
        for var, val in valores_ingresados.items():
            info = VARIABLES_INFO[var]
            if info["tipo"] in ("categorica", "binaria", "ordinal"):
                display_val = info["opciones"].get(val, val)
            else:
                display_val = f"{val} {info.get('unidad', '')}"
            st.markdown(
                f"**{NOMBRES_DISPLAY[var]}:** `{display_val}`"
            )
    else:
        st.caption("Ninguna variable seleccionada aún.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Resultado ──
    if predict_btn:
        if not variables_seleccionadas:
            st.error("Selecciona al menos una variable para predecir.")
        else:
            # Construir fila de datos completa con defaults para variables no seleccionadas
            fila = {}
            for var in TODAS_LAS_VARIABLES:
                if var in valores_ingresados:
                    fila[var] = valores_ingresados[var]
                else:
                    fila[var] = VARIABLES_INFO[var]["default"]

            df_input = pd.DataFrame([fila])

            with st.spinner("Calculando..."):
                prob = modelo.predict_proba(df_input)[0][1]
                prediccion = modelo.predict(df_input)[0]

            prob_pct = prob * 100
            riesgo_label = "ALTO RIESGO" if prediccion == 1 else "BAJO RIESGO"
            css_class = "result-high" if prediccion == 1 else "result-low"
            emoji = "⚠️" if prediccion == 1 else "✅"

            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-title">{emoji} {riesgo_label}</div>
                <div class="result-prob">{prob_pct:.1f}%</div>
                <div style="font-size:0.85rem; margin-top:6px; opacity:0.8;">
                    Probabilidad estimada de enfermedad cardiovascular
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Barra de probabilidad
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(prob, text=f"Confianza del modelo: {prob_pct:.1f}%")

            # Métricas rápidas
            st.markdown("**Detalle de la predicción**")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">P(Sin enfermedad)</div>
                    <div class="metric-value">{(1-prob)*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">P(Con enfermedad)</div>
                    <div class="metric-value">{prob_pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Aviso clínico
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(
                "⚕️ **Aviso:** Esta predicción es orientativa y no reemplaza el diagnóstico médico. "
                "Consulte siempre a un profesional de la salud."
            )

            # Exportar datos
            st.markdown("---")
            resultado_export = {**fila, "probabilidad_riesgo": round(prob, 4), "prediccion": int(prediccion)}
            df_export = pd.DataFrame([resultado_export])
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultado en CSV",
                data=csv,
                file_name=f"riesgo_{id_paciente}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Resultado")
        st.info("Completa los datos del paciente y presiona **Calcular predicción** para obtener el resultado.")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#94A3B8; font-size:0.8rem;'>"
    "Modelo Predictivo · Riesgo Cardiovascular · Uso exclusivo médico-investigativo"
    "</center>",
    unsafe_allow_html=True
)
