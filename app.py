import streamlit as st
from dotenv import load_dotenv
import time
import tempfile
from time import sleep
import numpy as np
from extraer_pdf import extract_text_from_pdf
from limpieza_texto import clean_text
from gemini_client import call_gemini
from deepseek_client import call_deepseek
from metricas import (
    semantic_relevance_score,
    distractor_quality_index,
    concept_coverage_semantic,
    question_diversity
)


# Nuevos imports para tablas
from extraer_tabla import extraer_tablas  # devuelve CSV como str
from gemini_client_analyser import call_gemini_analyzer  # interpreta CSV

# Carga variables de entorno (.env)
load_dotenv()

st.set_page_config(
    page_title="Procesador de PDF Inteligente",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Procesador de PDF Inteligente")
st.markdown("---")

st.subheader("📤 Subir Archivo PDF")
uploaded_file = st.file_uploader("Selecciona un archivo PDF:", type=["pdf"])

if uploaded_file:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Extraer texto
    status_text.text("🔄 Extrayendo texto del PDF...")
    progress_bar.progress(10)
    raw_text = extract_text_from_pdf(uploaded_file)

    # 2. Limpiar texto
    status_text.text("🧹 Limpiando y formateando texto...")
    progress_bar.progress(30)
    cleaned_text = clean_text(raw_text)

    # 3. Extraer tablas y obtener CSV
    status_text.text("📊 Extrayendo tablas del PDF...")
    progress_bar.progress(45)
    # Reiniciar puntero del archivo antes de la siguiente lectura
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        table_csv = extraer_tablas(tmp.name)

    # 4. Interpretación de tablas con Gemini
    status_text.text("🔍 Interpretando tablas con Gemini...")
    progress_bar.progress(60)
    table_summary = call_gemini_analyzer(table_csv)
    # Eliminar líneas vacías y asteriscos
    table_summary = [line.replace('*', '').strip() for line in table_summary if line.strip()]

    # 5. Concatenar interpretación al texto limpio
    enhanced_text = cleaned_text + "\n\n" + "\n".join(table_summary)

    # 6. Ideas principales generadas por Gemini (texto + resumen de tablas)
    status_text.text("💡 Extrayendo ideas principales...")
    progress_bar.progress(75)
    ideas = call_gemini(enhanced_text)

    # 7. Generar preguntas con DeepSeek
    status_text.text("❓ Generando preguntas con DeepSeek...")
    progress_bar.progress(90)
    questions = call_deepseek(ideas, num_questions=len(ideas))

    # 8. Obtener respuestas
    answers = [q.get('correct_answer', '') for q in questions]

    # Fin del procesamiento
    progress_bar.progress(100)
    status_text.text("🎉 ¡Procesamiento completado!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")

    # Mostrar resultados finales
    # Texto limpio
    with st.expander("🧹 Texto Limpio", expanded=True):
        st.text_area("Texto Limpio", cleaned_text, height=200, label_visibility="collapsed")

    # CSV de tablas
    with st.expander("📑 Tablas Encontradas", expanded=False):
        st.text_area("Tablas Encontradas", table_csv, height=200, label_visibility="collapsed")

    # Interpretación de tablas
    with st.expander("🗒️ Interpretación de Tablas", expanded=False):
        for i, line in enumerate(table_summary, 1):
            st.markdown(f"**{i}.** {line}")

    # Ideas principales finales
    with st.expander("💡 Ideas Principales", expanded=False):
        for i, idea in enumerate(ideas, 1):
            st.markdown(f"**{i}.** {idea}")

    # Preguntas generadas
    with st.expander("❓ Preguntas Generadas", expanded=False):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q['question']}")
            for opt in q.get('options', []):
                st.markdown(f"- {opt}")

    # Respuestas
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
        cov = concept_coverage_semantic(ideas, preguntas_text, top_k=25, threshold=0.4)
        div = question_diversity(preguntas_text)

        st.metric("Relevancia semántica promedio", f"{rel:.2f}")
        st.metric("Índice calidad distractores", f"{idx_d:.2f}")
        st.metric("Cobertura de conceptos (%)", f"{cov:.1f}%")
        st.metric("Diversidad de preguntas", f"{div:.2f}")

else:
    st.info("👆 Por favor, sube un archivo PDF para comenzar.")