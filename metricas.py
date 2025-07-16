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
def concept_coverage(text: str, questions: list[str], top_k: int = 20) -> float:
    """
    Extrae las top_k keywords de text via TF-IDF y retorna el % cubierto por las preguntas.
    """
    spanish_stopwords = stopwords.words('spanish')
    vectorizer = TfidfVectorizer(max_features=top_k, stop_words=spanish_stopwords)
    
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    covered = 0
    for kw in keywords:
        for q in questions:
            if SequenceMatcher(None, kw, q).ratio() > 0.8:
                covered += 1
                break
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