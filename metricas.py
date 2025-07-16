import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Inicializa el modelo de embeddings local (usa un modelo ligero preentrenado)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Relevancia semántica promedio
def semantic_relevance_score(text: str, questions: list[str]) -> float:
    """
    Calcula la similitud coseno media entre el embedding del texto completo
    y los embeddings de cada pregunta, usando sentence-transformers.
    """
    # Generar embedding del texto fuente
    emb_text = model.encode(text, convert_to_numpy=True)
    emb_text = emb_text.reshape(1, -1)
    # Generar embeddings de las preguntas
    emb_qs = model.encode(questions, convert_to_numpy=True)
    # Calcular similitudes
    sims = cosine_similarity(emb_qs, emb_text).flatten()
    return float(np.mean(sims))

# 2. Índice de calidad de distractores (0-1)
def distractor_quality_index(correct: str, distractors: list[str]) -> float:
    """
    Devuelve 1 - media(similitud_coseno(correct, cada distractor)).
    """
    # Embedding de la respuesta correcta
    emb_corr = model.encode(correct, convert_to_numpy=True).reshape(1, -1)
    # Embeddings de los distractores
    emb_d = model.encode(distractors, convert_to_numpy=True)
    sims = cosine_similarity(emb_corr, emb_d).flatten()
    return float(1 - np.mean(sims))

# 3. Cobertura de conceptos (%)
def concept_coverage_semantic(text: str | list[str], questions: list[str],
                              top_k: int = 20,
                              threshold: float = 0.5) -> float:
    """
    TF-IDF extrae top_k keywords, luego mide para
    cada keyword si existe alguna pregunta cuya
    similitud coseno >= threshold.
    """
    if isinstance(text, list):
        text = " ".join(text)
    if not text.strip() or not questions:
        return 0.0

    # 1) Extraer keywords
    spanish_sw = stopwords.words('spanish')
    vec = TfidfVectorizer(max_features=top_k, stop_words=spanish_sw)
    kw_matrix = vec.fit_transform([text])
    keywords = vec.get_feature_names_out()
    if len(keywords) == 0:
        return 0.0

    # 2) Embeddings de keywords y preguntas
    kw_emb = model.encode(keywords, convert_to_numpy=True)
    q_emb  = model.encode(questions, convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    # 3) Cuentas cuántas keywords superan el umbral
    covered = 0
    for kw_vec in kw_emb:
        sims = cosine_similarity([kw_vec], q_emb).flatten()
        if np.any(sims >= threshold):
            covered += 1

    return covered / len(keywords) * 100


# 4. Diversidad de preguntas (0-1)
def question_diversity(questions: list[str]) -> float:
    """
    Calcula la diversidad media entre embeddings de preguntas.
    """
    emb_qs = model.encode(questions, convert_to_numpy=True)
    n = emb_qs.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - cosine_similarity(emb_qs[i:i+1], emb_qs[j:j+1])[0,0]
            dists.append(d)
    return float(np.mean(dists))