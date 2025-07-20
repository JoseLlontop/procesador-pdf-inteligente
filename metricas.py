import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import joblib # libreria para almacenar embeddings en disco
import os
import hashlib # Para crear identificadores unicos de texto
from functools import lru_cache
nltk.download('stopwords')

# Configuración global
CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Descargar stopwords solo si es necesario
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# Modelo ligero para embeddings
MODEL = SentenceTransformer('all-MiniLM-L6-v2') 
# para transformar textos en un vectores que representan la semantica

# Cache de embeddings para evitar recálculos
def get_embedding(text: str) -> np.ndarray:
    """Obtiene embedding con caché persistente"""
    hash_key = hashlib.md5(text.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
    
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    
    embedding = MODEL.encode(text, convert_to_numpy=True)
    joblib.dump(embedding, cache_path)
    return embedding
    #Crea un ID para cada texto con md5
    #Si calcaulamos el embeddin, lo carga del disco
    #Si no, lo calcula para usos futuros

    #Es lo mismo pero procesa para varios textos a la vez

def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Procesa embeddings por lotes con caché"""
    embeddings = []
    texts_to_process = []
    cache_paths = []
    
    for text in texts:
        hash_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
        cache_paths.append(cache_path)
        
        if os.path.exists(cache_path):
            embeddings.append(joblib.load(cache_path))
        else:
            texts_to_process.append(text)
    
    # Procesar solo los textos sin caché
    if texts_to_process:
        new_embeddings = MODEL.encode(texts_to_process, convert_to_numpy=True)
        for text, emb, path in zip(texts_to_process, new_embeddings, 
                                  [p for p in cache_paths if not os.path.exists(p)]):
            joblib.dump(emb, path)
            embeddings.append(emb)
    
    return np.array(embeddings)


# 1. Relevancia semántica promedio optimizada. Cuan relacionada estan las preguntas del texto limpio. 
# Lo ideal es cerca de 1

def semantic_relevance_score(text: str, questions: list[str]) -> float:
    """
    Calcula la similitud coseno media usando caché
    """
    if not text.strip() or not questions:
        return 0.0
    
    
    # Embedding del texto fuente
    emb_text = get_embedding(text).reshape(1, -1)
    
    # Embeddings de las preguntas
    emb_qs = get_embeddings_batch(questions)
    
    # Calcular similitudes vectorizadas y promediamos

    sims = cosine_similarity(emb_qs, emb_text).flatten()
    return float(np.mean(sims))

# 2. Índice de calidad de distractores optimizado. Cuan buenas son las opciones.
#  Ve la similitud entre correcta y distractor.
# Valor ideal entre 0,6-0,8. Mientras mas alto,son faciles de descartar. Mientras mas bajo, puede confundir. 
def distractor_quality_index(correct: str, distractors: list[str]) -> float:

    # Filtrar distractores válidos
    valid_distractors = [d for d in distractors if d != correct]
    if not valid_distractors:
        return 0.0
    
    # Embeddings de la respuesta correcta y distractores
    emb_corr = get_embedding(correct).reshape(1, -1)
    emb_d = get_embeddings_batch(valid_distractors)
    
    # Calcular similitudes vectorizadas
    sims = cosine_similarity(emb_corr, emb_d).flatten()
    return float(1 - np.mean(sims))


# 3. Cobertura de conceptos optimizada
@lru_cache(maxsize=32)
def get_keywords(text: str, top_k: int = 20) -> list[str]:
    """Extrae keywords con caché"""
    spanish_sw = stopwords.words('spanish')
    vec = TfidfVectorizer(max_features=top_k, stop_words=spanish_sw)
    try:
        vec.fit([text])
        return vec.get_feature_names_out().tolist()
    except:
        return []
    
# Procentaje de conceptos clave que aparecen en las preguntas

def concept_coverage_semantic(text: str | list[str], questions: list[str],
                              top_k: int = 20,
                              threshold: float = 0.5) -> float:

    if isinstance(text, list):
        text = " ".join(text)
    if not text.strip() or not questions:
        return 0.0

    # Obtener keywords con caché
    keywords = get_keywords(text, top_k)
    if not keywords:
        return 0.0

    # Embeddings de keywords y preguntas
    kw_emb = get_embeddings_batch(keywords)
    q_emb = get_embeddings_batch(questions)
    
    # para cada kw, verifica si hay preguntas similares
    
    # Calcular cobertura vectorizada
    sim_matrix = cosine_similarity(kw_emb, q_emb)

    # Calculamos porcentaje
    covered = np.any(sim_matrix >= threshold, axis=1).sum()
    return (covered / len(keywords)) * 100

# 4. Diversidad de preguntas optimizada. Mide cuan diferentes son las preguntas entre si.
# Valor entre 0,4-0,7 (si es muy alto, muy variado. Si es muy bajo, muy redundate)
def question_diversity(questions: list[str]) -> float:

    if len(questions) < 2:
        return 0.0
    
    emb_qs = get_embeddings_batch(questions)
    
    # Calcular matriz de similitud completa
    sim_matrix = cosine_similarity(emb_qs)
    
    # Obtener solo el triángulo superior sin la diagonal
    n = sim_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    diversities = 1 - sim_matrix[triu_indices]
    
    return float(np.mean(diversities))