import streamlit as st
from dotenv import load_dotenv
import time
import numpy as np

# Importa tus módulos existentes
from extraer_pdf import extract_text_from_pdf
from limpieza_texto import clean_text
from gemini_client import call_gemini
from deepseek_client import call_deepseek
from metricas import (
    semantic_relevance_score,
    distractor_quality_index,
    concept_coverage,
    question_diversity
)


# Carga variables de entorno (.env)
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Procesador de PDF Inteligente",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Título y descripción
st.title("Procesador de PDF Inteligente")
st.markdown("---")
st.markdown(
    """
    ### 🚀 Bienvenido al Procesador de PDF Inteligente

    Esta aplicación te permite:
    - 📤 **Subir archivos PDF** de forma segura
    - 🧹 **Limpiar y formatear** el texto extraído
    - 💡 **Identificar ideas principales** del contenido con Gemini
    - ❓ **Generar preguntas** con DeepSeek
    - ✅ **Obtener respuestas** de DeepSeek
    """
)
st.markdown("---")

# Carga de PDF
st.subheader("📤 Subir Archivo PDF")
uploaded_file = st.file_uploader(
    "Selecciona un archivo PDF para procesar:",
    type=["pdf"],
    help="Sube un archivo PDF para extraer y analizar su contenido."
)

if uploaded_file:
    # Barra de progreso y estado
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Extraer
    status_text.text("🔄 Extrayendo texto del PDF...")
    progress_bar.progress(10)
    raw_text = extract_text_from_pdf(uploaded_file)

    # 2. Limpiar
    status_text.text("🧹 Limpiando y formateando texto...")
    progress_bar.progress(30)
    cleaned_text = clean_text(raw_text)

    # 3. Ideas principales con Gemini
    status_text.text("💡 Extrayendo ideas principales...")
    progress_bar.progress(50)
    ideas = call_gemini(cleaned_text)

    # Mostrar ideas brevemente en sidebar
    with st.sidebar.expander("💡 Ideas Principales", expanded=False):
        for i, idea in enumerate(ideas, 1):
            st.markdown(f"**{i}.** {idea}")

    # 4. Generar preguntas con DeepSeek
    status_text.text("❓ Generando preguntas con DeepSeek...")
    progress_bar.progress(70)
    questions = call_deepseek(ideas, num_questions=len(ideas))

    # 5. Obtener respuestas (usando DeepSeek u otra función)
    status_text.text("✅ Generando respuestas con DeepSeek...")
    progress_bar.progress(90)
    # Asumimos que call_deepseek también puede devolver respuestas si se ajusta el prompt,
    # o bien reutilizamos preguntas para respuestas. Aquí lo simplificamos:
    answers = [q.get('correct_answer', '') for q in questions]

    # Completo
    progress_bar.progress(100)
    status_text.text("🎉 Procesamiento completado!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")

    # Sección 1: Texto limpio
    with st.expander("🧹 Texto Limpio", expanded=True):
        st.text_area(
            "Texto Limpio", cleaned_text, height=300, label_visibility="collapsed"
        )

    # Sección 2: Ideas principales
    with st.expander("💡 Ideas Principales", expanded=False):
        for i, idea in enumerate(ideas, 1):
            st.markdown(f"**{i}.** {idea}")

    # Sección 3: Preguntas generadas
    with st.expander("❓ Preguntas Generadas", expanded=False):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q['question']}")
            for opt in q.get('options', []):
                st.markdown(f"- {opt}")

    # Sección 4: Respuestas
    with st.expander("✅ Respuestas", expanded=False):
        for i, ans in enumerate(answers, 1):
            st.markdown(f"**{i}.** {ans}")

    # Métricas de calidad
    with st.expander("📈 Métricas de Calidad", expanded=True):
        # Preparar listas
        preguntas_text = [q['question'] for q in questions]
        distractores_list = [q['options'] for q in questions]

        rel = semantic_relevance_score(cleaned_text, preguntas_text)
        idx_d = np.mean([distractor_quality_index(c, d) for c, d in zip(answers, distractores_list)])
        cov = concept_coverage(cleaned_text, preguntas_text)
        div = question_diversity(preguntas_text)

        st.metric("Relevancia semántica promedio", f"{rel:.2f}")
        st.metric("Índice calidad distractores", f"{idx_d:.2f}")
        st.metric("Cobertura de conceptos (%)", f"{cov:.1f}%")
        st.metric("Diversidad de preguntas", f"{div:.2f}")

    # Botón para reiniciar
    if st.button("🔄 Procesar Otro Archivo"):
        st.experimental_rerun()
else:
    st.info("👆 Por favor, sube un archivo PDF para comenzar el análisis.")

    with st.expander("📖 ¿Cómo usar esta aplicación?", expanded=False):
        st.markdown(
            """
            1. 📤 Sube tu PDF
            2. ⏳ Espera mientras procesa
            3. 📊 Explora las secciones desplegables
            """
        )
