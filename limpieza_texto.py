import re
import unicodedata

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True
) -> str:
    """
    Limpia y normaliza el texto para NLP:
    - Elimina etiquetas HTML.
    - Normaliza tildes y diacríticos.
    - Reduce saltos de línea y espacios excesivos.
    - Convierte a minúsculas.
    - Elimina puntuación.
    """
    # 1. Quitar etiquetas HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Normalización Unicode (é → e, ñ → n, etc.)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # 3. Reducir saltos de línea excesivos y líneas en blanco con espacios
    #    - Colapsar múltiples saltos de línea (incluyendo espacios) a uno
    text = re.sub(r'\n[ \t]*\n[ \t\n]*', '\n', text)

    # 4. Reducir espacios múltiples
    text = re.sub(r"[ ]{2,}", " ", text)

    # 5. Pasar a minúsculas
    if lowercase:
        text = text.lower()

    # 6. Eliminar puntuación (dejando solo letras y espacios)
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    # 7. Recortar espacios en los extremos de cada línea y del texto completo
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    
    return text.strip()
